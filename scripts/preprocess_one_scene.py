"""
Preprocess One Scene — debug rapido su un singolo frame/tripla
================================================================
Processa una singola tripla di frame e salva i tensori preprocessati.
Verifica che la pipeline preprocessing funzioni prima del training.

Uso (RaDelft):
    python scripts/preprocess_one_scene.py \\
        --config configs/radelft.yaml \\
        --scene_path data/raw/radelft/Scene1 \\
        --frame_idx 10 \\
        --out_dir /tmp/debug_preprocess

Uso (RADIal):
    python scripts/preprocess_one_scene.py \\
        --config configs/radial.yaml \\
        --scene_path data/raw/radial \\
        --sample_id 100 \\
        --out_dir /tmp/debug_preprocess

Checklist di debug (verifica queste condizioni prima del training):
    [ ] radar_cube.npy shape == (6, R, A, E) e dtype == float32
    [ ] rad_map.npy    shape == (D, R, A) e dtype == float32
    [ ] lidar_occ.npy  shape == (R, A, E) e somma > 0 (non è tutta zero!)
    [ ] BEV lidar_occ mostra strutture coerenti (strade, veicoli)
    [ ] radar_cube canale 0 (I_t) ha range non degenere (no NaN, no tutto-zero)
    [ ] Percentuale occupancy lidar_occ è tra 0.01% e 5% (sparsa ma non vuota)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf


def process_radelft_scene(cfg, scene_path: Path, frame_idx: int, out_dir: Path):
    """Debug per una singola tripla RaDelft."""
    import scipy.io
    from src.datasets.radelft_dataset import (
        RaDelftDataset, _rotation_matrix_z,
        _AZIMUTH_OFFSET_DEG, _X_OFFSET_CM, _Y_OFFSET_CM,
        _POWER_NORM, _ELEV_NORM, _VEL_FFT_SIZE, _VEL_BIN_SIZE,
    )
    from src.alignment.voxelization import Voxelizer, radelft_default_axes

    cubes_dir = scene_path / "RadarCubes"
    rods_dir  = scene_path / "rosDS"
    lidar_dir = rods_dir / "rslidar_points_clean"

    # Costruisci voxelizer
    range_axis, az_axis, el_axis = radelft_default_axes()
    vox = Voxelizer(range_axis, az_axis, el_axis)
    E, R_ax, A = len(el_axis), len(range_axis), len(az_axis)
    D = _VEL_FFT_SIZE

    print(f"Grid: R={R_ax}, A={A}, E={E}, D={D}")

    # Crea istanza del dataset per usare i metodi
    ds = RaDelftDataset.__new__(RaDelftDataset)
    ds.range_axis     = range_axis
    ds.azimuth_axis   = az_axis
    ds.elevation_axis = el_axis
    ds.E = E; ds.R = R_ax; ds.A = A; ds.D = D
    ds.voxelizer = vox
    ds.norm_vel  = True
    ds.vel_max   = 5.0

    # Tripla di frame: [frame_idx-2, frame_idx-1, frame_idx]
    frame_tensors = []
    for offset in [2, 1, 0]:
        fidx = max(1, frame_idx - offset)
        frame_info = {
            "scene":      int(scene_path.name.replace("Scene", "")),
            "frame_idx":  fidx,
            "power_path": str(cubes_dir / f"Pow_Frame_{fidx}.mat"),
            "ele_path":   str(cubes_dir / f"Ele_Frame_{fidx}.mat"),
            "lidar_path": "",
        }
        t_rae2, rad_map = ds._load_single_frame(frame_info)
        frame_tensors.append(t_rae2)
        if offset == 0:
            rad_map_t = rad_map

    # Fudi i 3 frame in (R, A, E, 6)
    radar_cube_rae6 = np.concatenate(
        [frame_tensors[2], frame_tensors[1], frame_tensors[0]], axis=-1
    )

    # Trova il LiDAR più vicino al frame_idx
    import os
    lidar_files = sorted(os.listdir(str(lidar_dir)))
    if lidar_files:
        lidar_path = str(lidar_dir / lidar_files[min(frame_idx, len(lidar_files)-1)])
        pts = np.load(lidar_path).reshape(-1, 3).astype(np.float64)
        R_mat = _rotation_matrix_z(_AZIMUTH_OFFSET_DEG)
        pts_rot = pts @ R_mat.T
        pts_rot[:, 0] += _X_OFFSET_CM / 100.0
        pts_rot[:, 1] += _Y_OFFSET_CM / 100.0
        lidar_occ = vox.voxelize(pts_rot)
    else:
        lidar_occ = np.zeros((R_ax, A, E), dtype=np.float32)

    return radar_cube_rae6, rad_map_t, lidar_occ


def process_radial_scene(cfg, root_path: Path, sample_id: int, out_dir: Path):
    """Debug per un singolo sample RADIal."""
    from src.datasets.radial_dataset import RADIalDataset
    from src.alignment.voxelization import Voxelizer

    ds = RADIalDataset(root_dir=str(root_path), mode="train", processing_mode="fft_only")
    idx = ds.sample_ids.index(sample_id) if sample_id in ds.sample_ids else 0
    sample = ds[idx]

    return (
        sample["radar_cube"].numpy().transpose(1, 2, 3, 0),  # (R, A, E, 6)
        sample["rad_map"].numpy(),                           # (D, R, A) or similar
        sample["lidar_occ"].numpy(),                         # (R, A, E)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--scene_path", default=None,
                        help="Path alla scena (per RaDelft) o alla root (per RADIal)")
    parser.add_argument("--frame_idx",  type=int, default=10, help="Frame/sample index")
    parser.add_argument("--sample_id",  type=int, default=None, help="Sample ID RADIal (override frame_idx)")
    parser.add_argument("--out_dir",    default="/tmp/debug_preprocess")
    parser.add_argument("--no_viz",     action="store_true")
    args = parser.parse_args()

    cfg     = OmegaConf.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.dataset.name
    scene_path   = Path(args.scene_path) if args.scene_path else Path(cfg.dataset.raw_path)

    print(f"\n{'='*60}")
    print(f"Debug preprocessing: {dataset_name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    if dataset_name == "radelft":
        radar_cube, rad_map, lidar_occ = process_radelft_scene(
            cfg, scene_path, args.frame_idx, out_dir
        )
    elif dataset_name == "radial":
        sid = args.sample_id or args.frame_idx
        radar_cube, rad_map, lidar_occ = process_radial_scene(cfg, scene_path, sid, out_dir)
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")

    # Salva
    np.save(out_dir / "radar_cube.npy", radar_cube)
    np.save(out_dir / "rad_map.npy",    rad_map)
    np.save(out_dir / "lidar_occ.npy",  lidar_occ)
    np.savez(out_dir / "sample.npz",
             radar_cube=radar_cube, rad_map=rad_map, lidar_occ=lidar_occ)

    # Statistiche
    from src.utils.visualization import print_preprocessing_stats
    print_preprocessing_stats(lidar_occ, radar_cube, rad_map)

    print(f"\n✓ Checklist:")
    print(f"  radar_cube  shape={radar_cube.shape}  dtype={radar_cube.dtype}")
    print(f"  rad_map     shape={rad_map.shape}     dtype={rad_map.dtype}")
    print(f"  lidar_occ   shape={lidar_occ.shape}   somma={lidar_occ.sum():.0f}"
          f"  ({100*lidar_occ.mean():.3f}% occupancy)")

    if lidar_occ.sum() == 0:
        print("  ⚠️  ATTENZIONE: lidar_occ è tutta zero! Verifica la pipeline LiDAR.")
    if np.isnan(radar_cube).any():
        print("  ⚠️  ATTENZIONE: radar_cube contiene NaN!")
    if not args.no_viz:
        from src.utils.visualization import plot_bev_occupancy, plot_radar_cube_channels
        plot_bev_occupancy(lidar_occ, title="LiDAR Occupancy GT",
                           save_path=out_dir / "bev_lidar.png")
        plot_radar_cube_channels(radar_cube.transpose(3, 0, 1, 2),
                                 save_path=out_dir / "radar_channels.png")
        print(f"\nVisualizzazioni salvate in {out_dir}")

    print(f"\nFile salvati in {out_dir}")


if __name__ == "__main__":
    main()
