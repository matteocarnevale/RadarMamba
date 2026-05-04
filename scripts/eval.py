"""
Evaluation — Radar-Mamba
=========================
Valuta un checkpoint sul test set e riporta:
  CD, HD, UCD, UHD, N_pred_points (Tabella 1 del paper)

Uso:
    python scripts/eval.py --config configs/radelft.yaml --checkpoint checkpoints/best.pth
    python scripts/eval.py --config configs/radial.yaml  --checkpoint checkpoints/best.pth --save_viz /tmp/viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.radar_mamba_unet import RadarMambaUNet
from src.losses.focal_loss import FocalLoss
from src.utils.metrics import PointCloudMetrics
from src.utils.visualization import plot_bev_comparison

# Importa factory dallo script di training
from scripts.train import build_dataset, build_model, axes_for_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--save_viz",   default=None, help="Directory per le visualizzazioni BEV")
    parser.add_argument("--n_viz",      type=int, default=5)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Modello
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint caricato: epoch={ckpt.get('epoch', '?')}, "
          f"best_CD={ckpt.get('best_cd', 'N/A')}")

    # Dataset test
    test_ds = build_dataset(cfg, "test")
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # Assi fisici per le metriche
    range_axis, az_axis, el_axis = axes_for_dataset(cfg)

    # Viz
    viz_dir = Path(args.save_viz) if args.save_viz else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    metrics = PointCloudMetrics()
    criterion = FocalLoss(
        alpha=cfg.training.focal_loss.alpha,
        gamma=cfg.training.focal_loss.gamma,
    )

    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            radar_cube = batch["radar_cube"].to(device)
            rad_map    = batch["rad_map"].to(device)
            gt_occ     = batch["lidar_occ"].to(device)

            logits = model(radar_cube, rad_map)
            loss   = criterion(logits, gt_occ)
            total_loss += loss.item()

            pred_prob = torch.sigmoid(logits)[0].cpu().numpy()   # (R, A, E)
            gt_np     = gt_occ[0].cpu().numpy()                  # (R, A, E)

            # Point cloud predetto e GT
            pred_occ = pred_prob > args.threshold
            ir_p, ia_p, ie_p = np.where(pred_occ)
            ir_g, ia_g, ie_g = np.where(gt_np > 0.5)

            pred_pts = np.stack([
                range_axis[np.clip(ir_p, 0, len(range_axis)-1)],
                az_axis[np.clip(ia_p, 0, len(az_axis)-1)],
                el_axis[np.clip(ie_p, 0, len(el_axis)-1)],
            ], axis=-1).astype(np.float32) if len(ir_p) else np.zeros((0,3), dtype=np.float32)

            gt_pts = np.stack([
                range_axis[np.clip(ir_g, 0, len(range_axis)-1)],
                az_axis[np.clip(ia_g, 0, len(az_axis)-1)],
                el_axis[np.clip(ie_g, 0, len(el_axis)-1)],
            ], axis=-1).astype(np.float32) if len(ir_g) else np.zeros((0,3), dtype=np.float32)

            metrics.update(pred_pts, gt_pts)

            if viz_dir and i < args.n_viz:
                plot_bev_comparison(
                    lidar_occ   = gt_np,
                    radar_pred  = pred_occ.astype(np.float32),
                    title       = f"Sample {i}  N_pred={len(ir_p)}  N_gt={len(ir_g)}",
                    save_path   = viz_dir / f"sample_{i:04d}.png",
                )

    result = metrics.compute()
    n = len(loader)
    print("\n" + "=" * 65)
    print("RISULTATI VALUTAZIONE")
    print("=" * 65)
    print(f"  Loss:              {total_loss/n:.4f}")
    print(f"  CD  (↓):           {result['CD']:.4f}")
    print(f"  HD  (↓):           {result['HD']:.4f}")
    print(f"  UCD (↓):           {result['UCD']:.4f}")
    print(f"  UHD (↓):           {result['UHD']:.4f}")
    print(f"  N_pred_pts (↑):    {result['n_pred_points']:.0f}")
    print("=" * 65)

    print("\nRiferimento Tabella 1 (RADIal):")
    print("  Radar-Mamba: CD=0.531, HD=13.602, UCD=0.429, UHD=13.334, N_pts=16963")
    print("\nRiferimento Tabella 1 (RaDelft):")
    print("  Radar-Mamba: CD=0.644, HD=15.829, UCD=0.393, UHD=15.468, N_pts=115025")


if __name__ == "__main__":
    main()
