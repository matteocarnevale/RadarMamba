"""
Offline preprocessing — RADIal
==============================
Converts RADIal `fft_{:06d}.npy` files into `.npz` tensors ready for training,
so AoA beamforming does not need to run at every epoch.

Per-frame pipeline (sample_id):
  1. Load `fft_{sample_id:06d}.npy` → `RD_spectrums` (512, 256, 16) complex
  2. MIMO assembly with DDMA Doppler-shift compensation (rpl.py logic)
  3. 3D AoA beamforming via `CalibrationTable.npy` → cube (n_az, n_el, R, D)
  4. Crop to the RADIal FoV: [R=480, A=736, E=11]
  5. Build tensor [R, A, E, 2] (intensity + velocity)
  6. Build RAD map [R, A, D_red=16]
  7. Load LiDAR point cloud `pcl_{sample_id:06d}.npy`
  8. Ground removal (Patchwork++) + voxelization → `lidar_occ` [R, A, E]
  9. Save everything into `processed/{sample_id:06d}.npz`

Temporal fusion (t-1, t-2) is handled by the Dataset which loads 3 consecutive
`.npz` files — this script does NOT pre-fuse them.

Usage:
    # Full preprocessing (all frames)
    python scripts/preprocess_radial.py --config configs/radial.yaml

    # Debug on the first N frames
    python scripts/preprocess_radial.py --config configs/radial.yaml --n_samples 10

    # Force recompute even if `.npz` already exists
    python scripts/preprocess_radial.py --config configs/radial.yaml --overwrite

    # Parallel CPU workers
    python scripts/preprocess_radial.py --config configs/radial.yaml --workers 8

Output structure in `processed_path/`:
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

# Limit OpenBLAS/MKL to 1 thread per process — must happen BEFORE importing NumPy.
# Without this, each matmul (CalibMat @ MIMO_r.T) can spawn N_CPU threads → CPU at 1400%+
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",   "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


# ── RADIal hardware parameters (from rpl.py) ──────────────────────
N_SAMPLES   = 512
N_CHIRPS    = 256
N_RX_ANT    = 16
N_TX_ANT    = 12
N_RED_DOP   = 16   # chirps per TX in DDMA (256/16 = 16 gruppi)
N_CPL       = 16   # chirps per loop

# RADIal target grid (paper Section 4.1)
GRID_R = 480
GRID_A = 736
GRID_E = 11
AZ_MIN_DEG = -75.0
AZ_MAX_DEG =  75.0
EL_MIN_DEG = -4.0
EL_MAX_DEG =  6.0
RANGE_SCALE = 103.0   # m (range max del chirp RADIal)

# Official test split (loader.py)
_TEST_SEQS = {
    "RECORD@2020-11-22_12.45.05", "RECORD@2020-11-22_12.25.47",
    "RECORD@2020-11-22_12.03.47", "RECORD@2020-11-22_12.54.38",
    "RECORD@2020-11-22_12.49.56", "RECORD@2020-11-22_12.11.49",
    "RECORD@2020-11-22_12.28.47", "RECORD@2020-11-21_14.25.06",
}


# ── CalibrationTable loading ──────────────────────────────────────

class AoABeamformer:
    """
    3D AoA beamforming for RADIal via `CalibrationTable.npy`.

    This replicates the logic of `rpl.py::RadarSignalProcessing.__get_PCL`,
    extended to all range–Doppler bins (no CFAR gating — the "blue path").
    """

    def __init__(self, calib_path: str) -> None:
        mat = np.load(calib_path, allow_pickle=True).item()
        # mat['Signal']: (n_az, n_virtual, n_el)
        sig         = mat["Signal"]                         # (n_az, n_virt, n_el)
        self.n_az   = sig.shape[0]
        self.n_virt = sig.shape[1]
        self.n_el   = sig.shape[2]

        # 3D calibration matrix: (n_az*n_el, n_virt)
        self.CalibMat = np.rollaxis(sig, 2, 1).reshape(
            self.n_az * self.n_el, self.n_virt
        )

        # Hanning window to reduce sidelobes
        self.window = mat["H"][0]                           # (n_virt,)

        # Physical angle tables
        self.az_table = mat["Azimuth_table"]    # (n_az,) degrees
        self.el_table = mat["Elevation_table"]  # (n_el,) degrees

        # DDMA: dividend array for TX re-ordering
        self.dividend = np.arange(0, N_RED_DOP * N_CPL, N_RED_DOP)

        # Target grid (indices into the physical angle tables)
        az_target = np.linspace(AZ_MIN_DEG, AZ_MAX_DEG, GRID_A)
        el_target = np.linspace(EL_MIN_DEG, EL_MAX_DEG, GRID_E)
        self.az_idx = np.searchsorted(self.az_table, az_target).clip(0, self.n_az - 1)
        self.el_idx = np.searchsorted(self.el_table, el_target).clip(0, self.n_el - 1)

        # Target range scale (first GRID_R range bins)
        self.r_scale = RANGE_SCALE / N_SAMPLES

    def _build_doppler_seqs(self) -> list[np.ndarray]:
        """Precompute DDMA Doppler index sequences for each Doppler bin."""
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
        Convert `RD_spectrums` (512, 256, 16) complex into a [R, A, E, 2] tensor.

        Processes range-bin by range-bin to avoid OOM:
        - Batch approach: (512*256, 128) @ (128, n_az*n_el) → ~4 GB per worker → OOM
        - Per-R approach: (256, 128)     @ (128, n_az*n_el) → ~8 MB per step  → OK

        Args:
            RD_spectrums: (N_SAMPLES=512, N_CHIRPS=256, N_RX=16) complex64

        Returns:
            tensor_rae2 : (GRID_R, GRID_A, GRID_E, 2)  float32
            rad_map      : (GRID_R, GRID_A, N_RED_DOP)   float32
        """
        R_crop       = min(N_SAMPLES, GRID_R)
        dop_seqs     = self._build_doppler_seqs()   # list(256) of (8,) arrays
        vel_bin_size = 0.04

        tensor_rae2 = np.zeros((GRID_R, GRID_A, GRID_E, 2), dtype=np.float32)
        rad_map     = np.zeros((GRID_R, GRID_A, N_RED_DOP),  dtype=np.float32)

        for r in range(R_crop):
            # ── MIMO assembly for this range bin ───────────────────────
            # RD[r, dop_seqs, :] → (N_CHIRPS, 8, 16) → reshape → (256, 128)
            MIMO_r = RD_spectrums[r, dop_seqs, :].reshape(N_CHIRPS, -1)  # (256, 128)
            MIMO_r = (MIMO_r * self.window).astype(np.complex64)

            # ── 3D AoA beamforming ─────────────────────────────────────
            # CalibMat: (n_az*n_el, 128) @ MIMO_r.T: (128, 256)
            # → ASpec_r: (n_az*n_el, 256) → reshape (n_az, n_el, 256)
            ASpec_r = np.abs(
                self.CalibMat @ MIMO_r.T          # (n_az*n_el, N_CHIRPS)
            ).reshape(self.n_az, self.n_el, N_CHIRPS).astype(np.float32)

            # Select target FoV indices → (GRID_A, GRID_E, N_CHIRPS)
            cube_r = ASpec_r[
                np.ix_(self.az_idx, self.el_idx, np.arange(N_CHIRPS))
            ]   # (GRID_A, GRID_E, N_CHIRPS)

            # ── tensor_rae2[r]: max over Doppler ───────────────────────
            d_star    = np.argmax(cube_r, axis=-1)                    # (A, E)
            intensity = cube_r[
                np.arange(GRID_A)[:, None],
                np.arange(GRID_E)[None, :],
                d_star
            ]                                                         # (A, E)
            intensity = np.log1p(intensity)
            velocity  = (d_star.astype(np.float32) - N_CHIRPS / 2.0) * vel_bin_size
            velocity  = np.clip(velocity / 5.0, -1.0, 1.0)

            tensor_rae2[r, :, :, 0] = intensity
            tensor_rae2[r, :, :, 1] = velocity

            # ── rad_map[r]: max over E → reduce Doppler ───────────────
            # max over E → (GRID_A, N_CHIRPS), then group Doppler
            power_ra  = cube_r.max(axis=1)                            # (A, N_CHIRPS)
            power_red = power_ra.reshape(
                GRID_A, N_RED_DOP, N_CPL
            ).mean(axis=-1)                                           # (A, N_RED_DOP)
            rad_map[r] = power_red

        # Per-frame normalization (using the frame global max)
        mx_i = tensor_rae2[..., 0].max()
        if mx_i > 0:
            tensor_rae2[..., 0] /= mx_i

        mx_r = rad_map.max()
        if mx_r > 0:
            rad_map = np.log1p(rad_map) / np.log1p(mx_r)

        return tensor_rae2, rad_map.astype(np.float32)


# ── LiDAR preprocessing ───────────────────────────────────────────

def preprocess_lidar(
    lidar_path: str | Path,
    voxelizer,
) -> np.ndarray:
    """
    Load the Velodyne LiDAR point cloud, remove ground (Patchwork++), voxelize.

    RADIal LiDAR: `pcl_{:06d}.npy` — structured Velodyne format
    (x, y, z, intensity, ring, timestamp).

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

    # Ground removal with Patchwork++
    from src.alignment.ground_removal import GroundRemover
    remover = GroundRemover()
    clean, _ = remover.remove_ground_with_ego(xyz)

    if len(clean) == 0:
        return np.zeros((GRID_R, GRID_A, GRID_E), dtype=np.float32)

    return voxelizer.voxelize(clean[:, :3]).astype(np.float32)


# ── Worker for parallel processing ───────────────────────────────

def process_one_sample(args: dict) -> tuple[int, bool, str]:
    """
    Process a single sample and save the `.npz`.
    Returns (sample_id, success, error_msg).
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

        # Load RD_spectrums
        raw = np.load(fft_path, allow_pickle=True)
        if np.iscomplexobj(raw):
            RD = raw.astype(np.complex64)
        else:
            # If real+imag are concatenated on the last axis
            mid = raw.shape[-1] // 2
            RD = (raw[..., :mid] + 1j * raw[..., mid:]).astype(np.complex64)

        # Ensure shape (512, 256, 16)
        if RD.shape != (N_SAMPLES, N_CHIRPS, N_RX_ANT):
            # Try transposing if axes are permuted
            if set(RD.shape) == {N_SAMPLES, N_CHIRPS, N_RX_ANT}:
                # (N_RX, N_SAMPLES, N_CHIRPS) or another order: search a valid permutation
                for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
                    r = RD.transpose(perm)
                    if r.shape == (N_SAMPLES, N_CHIRPS, N_RX_ANT):
                        RD = r
                        break
            else:
                return sample_id, False, f"Unexpected shape: {RD.shape}"

        # AoA beamforming → tensor_rae2 + rad_map
        beamformer = AoABeamformer(calib_path)
        tensor_rae2, rad_map = beamformer.process(RD)

        # LiDAR → lidar_occ
        lidar_occ = preprocess_lidar(lidar_path, voxelizer)

        # Save
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


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for RADIal: FFT + AoA → .npz"
    )
    parser.add_argument("--config",    required=True,  help="configs/radial.yaml")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if the .npz file already exists")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    raw_path    = Path(cfg.dataset.raw_path)
    proc_path   = Path(cfg.dataset.processed_path)
    calib_path  = str(cfg.dataset.get("calib_table_path",
                      raw_path / "CalibrationTable.npy"))
    project_root = str(Path(__file__).resolve().parents[1])

    proc_path.mkdir(parents=True, exist_ok=True)

    # Check CalibrationTable
    if not Path(calib_path).exists():
        print(f"\n⚠️  CalibrationTable.npy non trovato: {calib_path}")
        print("   Scaricalo da: https://github.com/valeoai/RADIal/tree/main/SignalProcessing")
        print("   Oppure aggiorna calib_table_path in configs/radial.yaml")
        sys.exit(1)

    # Read labels.csv to obtain sample_ids and splits
    labels_path = raw_path / "labels.csv"
    if not labels_path.exists():
        print(f"⚠️  labels.csv non trovato in {raw_path}")
        # Fallback: scan radar_FFT/ to infer sample_ids
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

    # Build the job list
    jobs = []
    for sid in sample_ids:
        fft_path   = str(raw_path / "radar_FFT" / f"fft_{sid:06d}.npy")
        # RADIal ready-to-use: LiDAR files are pcl_*.npy (NOT laser_PCL_*.npy)
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

    # Run preprocessing (single-process or multiprocess)
    results = {"ok": 0, "skipped": 0, "error": 0}
    errors  = []

    if args.workers == 1:
        # Single process with progress bar
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

    # Save JSON index (sample_id → split)
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
