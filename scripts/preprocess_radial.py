"""
Preprocessing Offline — RADIal
================================
Converte i file `fft_{:06d}.npy` di RADIal in tensori .npz pronti per
il training, senza dover fare AoA beamforming ad ogni epoch.

Pipeline per ogni frame (sample_id):
  1. Carica fft_{sample_id:06d}.npy → RD_spectrums (512, 256, 16) complex
  2. MIMO assembly con DDMA Doppler shift compensation (rpl.py logic)
  3. AoA 3D beamforming via CalibrationTable.npy → cubo (n_az, n_el, R, D)
  4. Crop al FoV RADIal: [R=480, A=736, E=11]
  5. Build tensor [R, A, E, 2] (intensity + velocity)
  6. Build RAD map [R, A, D_red=16]
  7. Carica laser_PCL_{sample_id:06d}.npy → LiDAR Velodyne
  8. Ground removal (Patchwork++) + voxelizzazione → lidar_occ [R, A, E]
  9. Salva tutto in processed/{sample_id:06d}.npz

I frame precedenti (t-1, t-2) per la fusione temporale vengono gestiti
dal Dataset che carica 3 .npz consecutivi — NON li pre-fonde qui.

Uso:
    # Preprocessing completo (tutti i frame del dataset)
    python scripts/preprocess_radial.py --config configs/radial.yaml

    # Debug su N frame soltanto
    python scripts/preprocess_radial.py --config configs/radial.yaml --n_samples 10

    # Forza il ricalcolo anche se i file .npz esistono già
    python scripts/preprocess_radial.py --config configs/radial.yaml --overwrite

    # Processa in parallelo su K workers CPU
    python scripts/preprocess_radial.py --config configs/radial.yaml --workers 8

Struttura output in processed_path/:
    processed_path/
    ├── 000042.npz      # sample_id=42
    │   ├── tensor_rae2  (480, 736, 11, 2)  float32
    │   ├── rad_map      (480, 736, 16)     float32
    │   └── lidar_occ    (480, 736, 11)     float32
    ├── 000043.npz
    ...
    └── index.json      # mapping sample_id → split (train/test)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


# ── Parametri hardware RADIal (da rpl.py) ─────────────────────────
N_SAMPLES   = 512
N_CHIRPS    = 256
N_RX_ANT    = 16
N_TX_ANT    = 12
N_RED_DOP   = 16   # chirps per TX in DDMA (256/16 = 16 gruppi)
N_CPL       = 16   # chirps per loop

# Griglia target RADIal (paper Section 4.1)
GRID_R = 480
GRID_A = 736
GRID_E = 11
AZ_MIN_DEG = -75.0
AZ_MAX_DEG =  75.0
EL_MIN_DEG = -4.0
EL_MAX_DEG =  6.0
RANGE_SCALE = 103.0   # m (range max del chirp RADIal)

# Split test ufficiale (loader.py)
_TEST_SEQS = {
    "RECORD@2020-11-22_12.45.05", "RECORD@2020-11-22_12.25.47",
    "RECORD@2020-11-22_12.03.47", "RECORD@2020-11-22_12.54.38",
    "RECORD@2020-11-22_12.49.56", "RECORD@2020-11-22_12.11.49",
    "RECORD@2020-11-22_12.28.47", "RECORD@2020-11-21_14.25.06",
}


# ── Caricamento CalibrationTable ──────────────────────────────────

class AoABeamformer:
    """
    AoA 3D beamforming per RADIal via CalibrationTable.npy.
    Replica la logica di rpl.py::RadarSignalProcessing.__get_PCL
    estesa a tutti i range-Doppler bins (senza gate CFAR — blue path).
    """

    def __init__(self, calib_path: str) -> None:
        mat = np.load(calib_path, allow_pickle=True).item()
        # mat['Signal']: (n_az, n_virtual, n_el)
        sig         = mat["Signal"]                         # (n_az, n_virt, n_el)
        self.n_az   = sig.shape[0]
        self.n_virt = sig.shape[1]
        self.n_el   = sig.shape[2]

        # Matrice di calibrazione 3D: (n_az*n_el, n_virt)
        self.CalibMat = np.rollaxis(sig, 2, 1).reshape(
            self.n_az * self.n_el, self.n_virt
        )

        # Finestra Hanning per ridurre sidelobes
        self.window = mat["H"][0]                           # (n_virt,)

        # Tabelle angoli fisici
        self.az_table = mat["Azimuth_table"]    # (n_az,) gradi
        self.el_table = mat["Elevation_table"]  # (n_el,) gradi

        # DDMA: dividend array per riordinamento TX
        self.dividend = np.arange(0, N_RED_DOP * N_CPL, N_RED_DOP)

        # Griglia target (indici negli angoli fisici)
        az_target = np.linspace(AZ_MIN_DEG, AZ_MAX_DEG, GRID_A)
        el_target = np.linspace(EL_MIN_DEG, EL_MAX_DEG, GRID_E)
        self.az_idx = np.searchsorted(self.az_table, az_target).clip(0, self.n_az - 1)
        self.el_idx = np.searchsorted(self.el_table, el_target).clip(0, self.n_el - 1)

        # Indici range target (primi GRID_R range bins, scalati)
        self.r_scale = RANGE_SCALE / N_SAMPLES

    def _build_doppler_seqs(self) -> list[np.ndarray]:
        """Pre-calcola le sequenze Doppler DDMA per ogni bin."""
        seqs = []
        for dop in range(N_CHIRPS):
            s = np.remainder(dop + self.dividend, N_CHIRPS)
            s = np.concatenate([[s[0]], s[5:]]).astype(int)  # 8 bin per TX
            seqs.append(s)
        return seqs

    def process(
        self, RD_spectrums: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converte RD_spectrums (512, 256, 16) complex nel tensore [R,A,E,2].

        Args:
            RD_spectrums: (N_SAMPLES=512, N_CHIRPS=256, N_RX=16) complex64
                          Output della 2D FFT (range × Doppler) per ogni antenna RX.

        Returns:
            tensor_rae2 : (GRID_R, GRID_A, GRID_E, 2)  float32
            rad_map      : (GRID_R, GRID_A, N_RED_DOP)   float32
        """
        R_crop = min(N_SAMPLES, GRID_R)

        # ── 1. MIMO assembly: riordiamo le sequenze Doppler DDMA ──────
        # Costruisce MIMO_spectrum (N_S*N_D, n_virt) applicando la
        # compensazione di fase TX per ogni bin range-Doppler.
        # REF: rpl.py::__get_PCL passi 3-4
        dop_seqs = self._build_doppler_seqs()   # list(256) di array(8,)
        # RD[:, dop_seqs, :] ha shape (N_S, N_D, 8, N_RX)
        # Reshape a (N_S*N_D, 8*N_RX=128)
        MIMO = RD_spectrums[:R_crop, dop_seqs, :].reshape(
            R_crop * N_CHIRPS, -1
        )                                        # (R_crop*256, 128)
        MIMO = MIMO * self.window                # windowing

        # ── 2. AoA 3D beamforming: CalibMat @ MIMO.T ─────────────────
        # Output: (n_az*n_el, R_crop*256) → reshape a (n_az, n_el, R_crop, 256)
        ASpec = np.abs(
            self.CalibMat @ MIMO.T                   # (n_az*n_el, R_crop*256)
        ).reshape(self.n_az, self.n_el, R_crop, N_CHIRPS)
        # Seleziona gli indici angolari del FoV target
        # ASpec[az_idx, el_idx, :, :] → (GRID_A, GRID_E, R_crop, N_CHIRPS)
        cube_ae = ASpec[
            np.ix_(self.az_idx, self.el_idx,
                   np.arange(R_crop), np.arange(N_CHIRPS))
        ]   # (GRID_A, GRID_E, R_crop, N_CHIRPS)

        # Trasponi → (R_crop, GRID_A, GRID_E, N_CHIRPS)
        cube = cube_ae.transpose(2, 0, 1, 3).astype(np.float32)

        # Pad se R_crop < GRID_R
        if R_crop < GRID_R:
            pad = np.zeros((GRID_R - R_crop, GRID_A, GRID_E, N_CHIRPS), np.float32)
            cube = np.concatenate([cube, pad], axis=0)

        # ── 3. tensor_rae2: intensity=max_D, velocity=argmax_D→m/s ───
        d_star    = np.argmax(cube, axis=-1)                 # (R, A, E)
        intensity = cube[
            np.arange(GRID_R)[:, None, None],
            np.arange(GRID_A)[None, :, None],
            np.arange(GRID_E)[None, None, :],
            d_star
        ]                                                     # (R, A, E)

        # Log-scale + normalizzazione per-frame
        intensity = np.log1p(intensity)
        mx = intensity.max()
        if mx > 0:
            intensity /= mx

        # Velocità: argmax Doppler → m/s (approx., velocità bin size ≈ 0.04 m/s)
        vel_bin_size = 0.04
        velocity = (d_star.astype(np.float32) - N_CHIRPS / 2.0) * vel_bin_size
        velocity = np.clip(velocity / 5.0, -1.0, 1.0)       # normalizza [-1, 1]

        tensor_rae2 = np.stack([intensity, velocity], axis=-1)  # (R, A, E, 2)

        # ── 4. RAD map: potenza aggregata su D ridotti ─────────────────
        # power su tutto (R, A, E, D) → somma su E → (R, A, D)
        # Poi riduciamo D=256 in D_red=16 sommando per gruppo
        power_rae = cube.sum(axis=2)                          # (R, A, N_CHIRPS)
        power_red = power_rae.reshape(
            GRID_R, GRID_A, N_RED_DOP, N_CPL
        ).mean(axis=-1)                                       # (R, A, N_RED_DOP=16)

        # Normalizza
        mx2 = power_red.max()
        if mx2 > 0:
            power_red = np.log1p(power_red) / np.log1p(mx2)

        return tensor_rae2.astype(np.float32), power_red.astype(np.float32)


# ── LiDAR preprocessing ───────────────────────────────────────────

def preprocess_lidar(
    lidar_path: str | Path,
    voxelizer,
) -> np.ndarray:
    """
    Carica il LiDAR Velodyne, rimuove il suolo (Patchwork++), voxelizza.

    RADIal LiDAR: laser_PCL_{:06d}.npy — formato Velodyne strutturato (x,y,z,i,ring,ts)
    REF: FFTRadNet/dataset/dataset.py mode='lidar'

    Returns:
        lidar_occ: (GRID_R, GRID_A, GRID_E) float32
    """
    from numpy.lib.recfunctions import structured_to_unstructured

    lidar_path = Path(lidar_path)
    if not lidar_path.exists():
        return np.zeros((GRID_R, GRID_A, GRID_E), dtype=np.float32)

    raw = np.load(str(lidar_path))
    if raw.dtype.names:                          # array strutturato
        pc = structured_to_unstructured(raw).reshape(-1, len(raw.dtype.names))
    else:
        pc = raw.reshape(-1, max(raw.shape[-1] if raw.ndim > 1 else 6, 3))
    xyz = pc[:, :3].astype(np.float64)

    if len(xyz) == 0:
        return np.zeros((GRID_R, GRID_A, GRID_E), dtype=np.float32)

    # Ground removal con Patchwork++
    from src.alignment.ground_removal import GroundRemover
    remover = GroundRemover()
    clean, _ = remover.remove_ground_with_ego(xyz)

    if len(clean) == 0:
        return np.zeros((GRID_R, GRID_A, GRID_E), dtype=np.float32)

    return voxelizer.voxelize(clean[:, :3]).astype(np.float32)


# ── Worker per processing parallelo ──────────────────────────────

def process_one_sample(args: dict) -> tuple[int, bool, str]:
    """
    Processa un singolo sample e salva il .npz.
    Ritorna (sample_id, success, error_msg).
    """
    sample_id   = args["sample_id"]
    fft_path    = args["fft_path"]
    lidar_path  = args["lidar_path"]
    out_path    = args["out_path"]
    calib_path  = args["calib_path"]
    overwrite   = args["overwrite"]

    if not overwrite and Path(out_path).exists():
        return sample_id, True, "skipped"

    try:
        import sys
        sys.path.insert(0, args["project_root"])

        from src.alignment.voxelization import Voxelizer
        voxelizer = Voxelizer.for_radial()

        # Carica RD_spectrums
        raw = np.load(fft_path, allow_pickle=True)
        if np.iscomplexobj(raw):
            RD = raw.astype(np.complex64)
        else:
            # Se real+imag concatenati sull'ultimo asse
            mid = raw.shape[-1] // 2
            RD = (raw[..., :mid] + 1j * raw[..., mid:]).astype(np.complex64)

        # Assicurati della shape (512, 256, 16)
        if RD.shape != (N_SAMPLES, N_CHIRPS, N_RX_ANT):
            # Prova a trasporre se gli assi sono invertiti
            if set(RD.shape) == {N_SAMPLES, N_CHIRPS, N_RX_ANT}:
                # (N_RX, N_SAMPLES, N_CHIRPS) o altro ordine
                # cerca la combinazione corretta
                for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
                    r = RD.transpose(perm)
                    if r.shape == (N_SAMPLES, N_CHIRPS, N_RX_ANT):
                        RD = r
                        break
            else:
                return sample_id, False, f"Shape inattesa: {RD.shape}"

        # AoA beamforming → tensor_rae2 + rad_map
        beamformer = AoABeamformer(calib_path)
        tensor_rae2, rad_map = beamformer.process(RD)

        # LiDAR → lidar_occ
        lidar_occ = preprocess_lidar(lidar_path, voxelizer)

        # Salva
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            tensor_rae2=tensor_rae2,   # (480, 736, 11, 2)
            rad_map=rad_map,            # (480, 736, 16)
            lidar_occ=lidar_occ,        # (480, 736, 11)
        )
        return sample_id, True, "ok"

    except Exception as e:
        return sample_id, False, traceback.format_exc(limit=3)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing offline RADIal: FFT + AoA → .npz"
    )
    parser.add_argument("--config",    required=True,  help="configs/radial.yaml")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Numero di sample da processare (None = tutti)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ricalcola anche se il file .npz esiste già")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Processi paralleli (default 4)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    raw_path    = Path(cfg.dataset.raw_path)
    proc_path   = Path(cfg.dataset.processed_path)
    calib_path  = str(cfg.dataset.get("calib_table_path",
                      raw_path / "CalibrationTable.npy"))
    project_root = str(Path(__file__).resolve().parents[1])

    proc_path.mkdir(parents=True, exist_ok=True)

    # Verifica CalibrationTable
    if not Path(calib_path).exists():
        print(f"\n⚠️  CalibrationTable.npy non trovato: {calib_path}")
        print("   Scaricalo da: https://github.com/valeoai/RADIal/tree/main/SignalProcessing")
        print("   Oppure aggiorna calib_table_path in configs/radial.yaml")
        sys.exit(1)

    # Leggi labels.csv per ottenere sample_ids e split
    labels_path = raw_path / "labels.csv"
    if not labels_path.exists():
        print(f"⚠️  labels.csv non trovato in {raw_path}")
        # Fallback: scansiona radar_FFT/ per trovare i sample_ids
        fft_dir = raw_path / "radar_FFT"
        if fft_dir.exists():
            sample_ids = sorted([
                int(f.stem.split("_")[-1])
                for f in fft_dir.glob("fft_*.npy")
            ])
            id_to_seq  = {}
        else:
            print(f"⚠️  Cartella radar_FFT/ non trovata in {raw_path}")
            sys.exit(1)
    else:
        df = pd.read_csv(str(labels_path))
        sample_col = df.columns[0]
        seq_col    = df.columns[14] if len(df.columns) > 14 else None
        sample_ids = sorted(df[sample_col].unique().tolist())
        id_to_seq  = {}
        if seq_col:
            for sid in sample_ids:
                rows = df[df[sample_col] == sid]
                id_to_seq[sid] = str(rows[seq_col].iloc[0])

    if args.n_samples:
        sample_ids = sample_ids[:args.n_samples]

    print(f"\n{'='*60}")
    print(f"RADIal Preprocessing Offline")
    print(f"{'='*60}")
    print(f"  Dataset:    {raw_path}")
    print(f"  Output:     {proc_path}")
    print(f"  CalibTable: {calib_path}")
    print(f"  Sample IDs: {len(sample_ids)} frame da processare")
    print(f"  Workers:    {args.workers}")
    print(f"{'='*60}\n")

    # Costruisci la lista di job
    jobs = []
    for sid in sample_ids:
        fft_path   = str(raw_path / "radar_FFT" / f"fft_{sid:06d}.npy")
        # RADIal ready-to-use: LiDAR files sono pcl_*.npy (NON laser_PCL_*.npy)
        lidar_path = str(raw_path / "laser_PCL" / f"pcl_{sid:06d}.npy")
        out_path   = str(proc_path / f"{sid:06d}.npz")

        if not Path(fft_path).exists():
            continue

        jobs.append({
            "sample_id":   sid,
            "fft_path":    fft_path,
            "lidar_path":  lidar_path,
            "out_path":    out_path,
            "calib_path":  calib_path,
            "overwrite":   args.overwrite,
            "project_root": project_root,
        })

    print(f"  File FFT trovati: {len(jobs)} / {len(sample_ids)}")
    if not jobs:
        print("⚠️  Nessun file fft_*.npy trovato. Verifica il path.")
        sys.exit(1)

    # Processa in parallelo
    results = {"ok": 0, "skipped": 0, "error": 0}
    errors  = []

    if args.workers == 1:
        # Single process con progress bar
        for job in tqdm(jobs, desc="Preprocessing"):
            sid, ok, msg = process_one_sample(job)
            if msg == "skipped":
                results["skipped"] += 1
            elif ok:
                results["ok"] += 1
            else:
                results["error"] += 1
                errors.append((sid, msg))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_sample, job): job for job in jobs}
            with tqdm(total=len(jobs), desc="Preprocessing") as pbar:
                for fut in as_completed(futures):
                    sid, ok, msg = fut.result()
                    if msg == "skipped":
                        results["skipped"] += 1
                    elif ok:
                        results["ok"] += 1
                    else:
                        results["error"] += 1
                        errors.append((sid, msg))
                    pbar.update(1)

    # Salva indice JSON (sample_id → split)
    index = []
    for sid in sample_ids:
        out_path = str(proc_path / f"{sid:06d}.npz")
        if Path(out_path).exists():
            seq   = id_to_seq.get(sid, "")
            split = "test" if seq in _TEST_SEQS else "train"
            index.append({"sample_id": sid, "split": split, "npz_path": out_path})

    index_path = proc_path / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETATO")
    print(f"  Processati:  {results['ok']}")
    print(f"  Saltati:     {results['skipped']}  (già esistenti)")
    print(f"  Errori:      {results['error']}")
    if errors:
        print(f"\n  Primi errori:")
        for sid, msg in errors[:3]:
            print(f"    sample {sid}: {msg[:200]}")
    print(f"\n  Indice salvato: {index_path}")
    print(f"  Train: {sum(1 for x in index if x['split']=='train')} sample")
    print(f"  Test:  {sum(1 for x in index if x['split']=='test')} sample")
    print(f"{'='*60}")
    print(f"\nPasso successivo:")
    print(f"  python scripts/train.py --config configs/radial.yaml")


if __name__ == "__main__":
    main()
