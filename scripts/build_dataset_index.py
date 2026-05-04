"""
Build Dataset Index
====================
Scansiona le cartelle raw e genera un indice JSON con la lista
di tutti i campioni (scene, frame, path, split).

Questo script è OPZIONALE: RaDelftDataset e RADIalDataset costruiscono
il loro indice interno automaticamente. Usalo per:
  - Ispezionare quanti campioni ci sono prima del training
  - Salvare un indice statico su disco per debug

Uso:
    python scripts/build_dataset_index.py --config configs/radelft.yaml
    python scripts/build_dataset_index.py --config configs/radial.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy.io
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


# ------------------------------------------------------------------
# RaDelft Index
# REF: RaDelft/loaders/rad_cube_loader.py::RADCUBE_DATASET_TIME
# ------------------------------------------------------------------

def build_radelft_index(cfg) -> list[dict]:
    """
    Scansiona le scene RaDelft e costruisce l'indice JSON.

    Struttura attesa:
        <raw_path>/Scene{N}/RadarCubes/Pow_Frame_{idx}.mat
        <raw_path>/Scene{N}/RadarCubes/timestamps.mat
        <raw_path>/Scene{N}/rosDS/rslidar_points_clean/{ts}.npy

    Split: Scene 2 e 6 = test, Scene 1,3,4,5,7 = train (paper Sec 4.1)
    """
    raw_path   = Path(cfg.dataset.raw_path)
    train_sc   = list(cfg.dataset.split.train_scenes)
    test_sc    = list(cfg.dataset.split.test_scenes)
    all_scenes = train_sc + test_sc

    def _get_timestamps_and_paths(directory: str) -> dict[int, str]:
        out = {}
        for fname in os.listdir(directory):
            if fname.endswith(".npy"):
                parts = fname.split(".")
                if len(parts) >= 2:
                    try:
                        ts = int(parts[0]) * 10**9 + int(parts[1])
                        out[ts] = os.path.join(directory, fname)
                    except ValueError:
                        pass
        return out

    def _closest_ts(new_ts, ts_dict):
        return min(ts_dict.keys(), key=lambda t: abs(t - new_ts))

    samples = []

    for scene_num in all_scenes:
        scene_dir = raw_path / f"Scene{scene_num}"
        cubes_dir = scene_dir / "RadarCubes"
        lidar_dir = scene_dir / "rosDS" / "rslidar_points_clean"

        if not cubes_dir.exists():
            print(f"  ⚠️  Scene{scene_num} non trovata in {cubes_dir} — saltata")
            continue

        # Trova tutti i frame disponibili
        pow_files = [f for f in os.listdir(str(cubes_dir)) if "Pow_Frame" in f]
        frame_nums = sorted([int(f.split("_")[-1].split(".")[0]) for f in pow_files])

        if not frame_nums:
            print(f"  ⚠️  Nessun Pow_Frame in Scene{scene_num} — saltata")
            continue

        # Carica il mapping timestamp → frame_num
        ts_mat_path = cubes_dir / "timestamps.mat"
        if ts_mat_path.exists():
            frame_num_to_ts = scipy.io.loadmat(str(ts_mat_path))["unixDateTime"]
        else:
            frame_num_to_ts = None
            print(f"  ⚠️  timestamps.mat non trovato per Scene{scene_num}")

        # Mapping timestamp → path LiDAR
        lidar_ts2path: dict[int, str] = {}
        if lidar_dir.exists():
            lidar_ts2path = _get_timestamps_and_paths(str(lidar_dir))

        split = "test" if scene_num in test_sc else "train"

        # Costruisci una entry per ogni frame (a partire dal frame 3 per avere t-2)
        for idx in frame_nums:
            if frame_num_to_ts is not None and idx <= len(frame_num_to_ts):
                ts_ns = int(frame_num_to_ts[idx - 1][0]) * 10**9
            else:
                ts_ns = idx   # fallback

            # Path LiDAR più vicino
            lidar_path = ""
            if lidar_ts2path:
                lt = _closest_ts(ts_ns, lidar_ts2path)
                lidar_path = lidar_ts2path[lt]

            samples.append({
                "scene_id":    scene_num,
                "frame_idx":   idx,
                "split":       split,
                "power_path":  str(cubes_dir / f"Pow_Frame_{idx}.mat"),
                "ele_path":    str(cubes_dir / f"Ele_Frame_{idx}.mat"),
                "lidar_path":  lidar_path,
                "timestamp_ns": ts_ns,
            })

    return samples


# ------------------------------------------------------------------
# RADIal Index
# REF: valeoai/RADIal/FFTRadNet/dataset/dataset.py
# REF: valeoai/RADIal/loader/loader.py
# ------------------------------------------------------------------

# Sequenze di test/val dallo split ufficiale (loader.py)
_RADIAL_TEST_SEQS = {
    "RECORD@2020-11-22_12.45.05",
    "RECORD@2020-11-22_12.25.47",
    "RECORD@2020-11-22_12.03.47",
    "RECORD@2020-11-22_12.54.38",
    "RECORD@2020-11-22_12.49.56",
    "RECORD@2020-11-22_12.11.49",
    "RECORD@2020-11-22_12.28.47",
    "RECORD@2020-11-21_14.25.06",
}


def build_radial_index(cfg) -> list[dict]:
    """
    Costruisce l'indice RADIal dal file labels.csv.

    Struttura attesa (dataset ready-to-use):
        <raw_path>/labels.csv
        <raw_path>/radar_FFT/fft_{:06d}.npy
        <raw_path>/laser_PCL/laser_PCL_{:06d}.npy   (opzionale)

    labels.csv colonne:
        0 = numSample (sample_id), 14 = dataset (sequence name), -1 = Difficult

    Split: le 8 sequenze in _RADIAL_TEST_SEQS → test, resto → train
    (81 train / 10 test come nel paper Radar-Mamba Section 4.1)
    """
    raw_path    = Path(cfg.dataset.raw_path)
    labels_path = raw_path / "labels.csv"
    fft_dir     = raw_path / "radar_FFT"
    lidar_dir   = raw_path / "laser_PCL"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.csv non trovato in {raw_path}\n"
            "Scarica il dataset RADIal ready-to-use da https://github.com/valeoai/RADIal"
        )

    df = pd.read_csv(str(labels_path))

    # Colonna 0 = sample_id, colonna 14 = nome sequenza (se esiste)
    sample_col = df.columns[0]
    seq_col    = df.columns[14] if len(df.columns) > 14 else None

    # Unique sample IDs
    unique_ids = sorted(df[sample_col].unique().tolist())

    # Mappa sample_id → sequence name
    id_to_seq = {}
    if seq_col is not None:
        for sid in unique_ids:
            rows = df[df[sample_col] == sid]
            id_to_seq[sid] = str(rows[seq_col].iloc[0])

    samples = []
    for sid in unique_ids:
        seq_name = id_to_seq.get(sid, "")
        split    = "test" if seq_name in _RADIAL_TEST_SEQS else "train"

        fft_path   = str(fft_dir   / f"fft_{sid:06d}.npy")
        lidar_path = str(lidar_dir / f"laser_PCL_{sid:06d}.npy")

        # Controlla che il file FFT esista
        if fft_dir.exists() and not Path(fft_path).exists():
            continue   # frame senza FFT → salta

        samples.append({
            "sample_id":   sid,
            "sequence":    seq_name,
            "split":       split,
            "fft_path":    fft_path,
            "lidar_path":  lidar_path if Path(lidar_path).exists() else "",
            # Frame precedenti per la fusione temporale (sample_id - 1, - 2)
            "fft_path_tm1": str(fft_dir / f"fft_{max(unique_ids[0], sid-1):06d}.npy"),
            "fft_path_tm2": str(fft_dir / f"fft_{max(unique_ids[0], sid-2):06d}.npy"),
        })

    return samples


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build dataset index JSON")
    parser.add_argument("--config",     required=True,  help="Path al config YAML")
    parser.add_argument("--output_dir", default=None,
                        help="Directory output (default: processed_path dal config)")
    args = parser.parse_args()

    cfg          = OmegaConf.load(args.config)
    dataset_name = cfg.dataset.name

    print(f"Building index for dataset : {dataset_name}")
    print(f"Raw data path             : {cfg.dataset.raw_path}")

    if dataset_name == "radial":
        samples = build_radial_index(cfg)
    elif dataset_name == "radelft":
        samples = build_radelft_index(cfg)
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}")

    # Separa split
    train_samples = [s for s in samples if s["split"] == "train"]
    test_samples  = [s for s in samples if s["split"] == "test"]

    print(f"\nTotale campioni : {len(samples)}")
    print(f"  Train         : {len(train_samples)}")
    print(f"  Test          : {len(test_samples)}")

    # Salva su disco
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.dataset.processed_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path   = out_dir / "index_all.json"
    train_path = out_dir / "index_train.json"
    test_path  = out_dir / "index_test.json"

    with open(all_path,   "w") as f: json.dump(samples,       f, indent=2)
    with open(train_path, "w") as f: json.dump(train_samples, f, indent=2)
    with open(test_path,  "w") as f: json.dump(test_samples,  f, indent=2)

    print(f"\nSalvati in {out_dir}:")
    print(f"  {all_path.name}")
    print(f"  {train_path.name}")
    print(f"  {test_path.name}")

    # Stampa le prime 2 entry per verifica
    if samples:
        print(f"\nPrima entry di esempio:")
        import pprint
        pprint.pprint(samples[0], indent=2)


if __name__ == "__main__":
    main()
