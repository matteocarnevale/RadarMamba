"""
Training Script — Radar-Mamba
================================
REF: Paper Section 4.2:
     AdamW, lr=0.001, cosine decay, 50 epochs, batch_size=16, 8×V100
     Focal loss α=0.995, γ=2

Usage:
    python scripts/train.py --config configs/radelft.yaml
    python scripts/train.py --config configs/radial.yaml --fast_dev_run
    python scripts/train.py --config configs/radelft.yaml --resume checkpoints/epoch_010.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Helps reduce allocator fragmentation for large 4D radar tensors.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.radar_mamba_unet import RadarMambaUNet, build_model_for_radial, build_model_for_radelft
from src.losses.focal_loss import FocalLoss
from src.utils.metrics import PointCloudMetrics


# ------------------------------------------------------------------
# Factory: build dataset + model from config
# ------------------------------------------------------------------

def build_dataset(cfg, split: str):
    name = cfg.dataset.name
    if name == "radelft":
        from src.datasets.radelft_dataset import RaDelftDataset
        return RaDelftDataset(
            dataset_path=cfg.dataset.raw_path,
            mode=split,
            normalize_velocity=True,
        )
    elif name == "radial":
        from src.datasets.radial_dataset import RADIalDataset
        # RADIal uses `.npz` preprocessed by preprocess_radial.py.
        # Run first: python scripts/preprocess_radial.py --config configs/radial.yaml
        return RADIalDataset(
            processed_path=cfg.dataset.processed_path,
            mode=split,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_model(cfg) -> RadarMambaUNet:
    name = cfg.dataset.name
    if name == "radelft":
        return build_model_for_radelft()
    elif name == "radial":
        return build_model_for_radial()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def axes_for_dataset(cfg):
    """
    Return (range_axis, az_axis, el_axis) used to convert occupancy → point cloud.
    These must match EXACTLY the axes used by the Dataset's voxelization.
    """
    name = cfg.dataset.name
    if name == "radelft":
        from src.alignment.voxelization import radelft_default_axes
        return radelft_default_axes()   # (500,) (240,) (34,) — non-uniform physical axes
    elif name == "radial":
        from src.alignment.voxelization import Voxelizer
        v = Voxelizer.for_radial()
        return v.range_axis, v.azimuth_axis, v.elevation_axis   # (480,) (736,) (11,)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_one_epoch(
    model:      RadarMambaUNet,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  FocalLoss,
    device:     torch.device,
    scaler:     torch.amp.GradScaler,
    epoch:      int,
    log_every:  int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch+1}", leave=False)
    for step, batch in enumerate(pbar):
        radar_cube = batch["radar_cube"].to(device, non_blocking=True)  # (B, 6, R, A, E)
        rad_map    = batch["rad_map"].to(device, non_blocking=True)     # (B, D, R, A)
        gt_occ     = batch["lidar_occ"].to(device, non_blocking=True)   # (B, R, A, E)

        try:
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(radar_cube, rad_map)                     # (B, R, A, E)
                loss   = criterion(logits, gt_occ)
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDNN_STATUS_NOT_INITIALIZED" in msg and device.type == "cuda":
                if torch.backends.cudnn.enabled:
                    warnings.warn(
                        "cuDNN failed to initialize in forward pass. "
                        "Disabling cuDNN globally and retrying this step.",
                        RuntimeWarning,
                    )
                    torch.backends.cudnn.enabled = False
                    torch.cuda.empty_cache()
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        logits = model(radar_cube, rad_map)
                        loss   = criterion(logits, gt_occ)
                else:
                    raise RuntimeError(
                        "cuDNN failed to initialize during forward pass even after fallback. "
                        "This is usually GPU memory pressure. "
                        "Lower `training.batch_size` (recommended: 2 on 1 GPU) and retry."
                    ) from exc
            else:
                raise

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:     RadarMambaUNet,
    loader:    DataLoader,
    criterion: FocalLoss,
    device:    torch.device,
    range_axis, azimuth_axis, elevation_axis,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    metrics    = PointCloudMetrics()

    import numpy as np
    from src.alignment.coordinate_transform import cartesian_to_spherical_rad

    for batch in tqdm(loader, desc="Eval", leave=False):
        radar_cube = batch["radar_cube"].to(device)
        rad_map    = batch["rad_map"].to(device)
        gt_occ     = batch["lidar_occ"].to(device)

        with torch.amp.autocast("cuda", enabled=False):
            logits = model(radar_cube, rad_map)
            loss   = criterion(logits, gt_occ)

        total_loss += loss.item()
        n_batches  += 1

        # Convert predictions and GT to point clouds for the metrics
        pred_pcs = model.predict_pointcloud(
            radar_cube, rad_map,
            range_axis, azimuth_axis, elevation_axis,
            threshold,
        )
        for b in range(gt_occ.shape[0]):
            gt_b = gt_occ[b].cpu().numpy()
            ir, ia, ie = np.where(gt_b > 0.5)
            if len(ir) == 0:
                continue
            gt_pts = np.stack([
                range_axis[np.clip(ir, 0, len(range_axis)-1)],
                azimuth_axis[np.clip(ia, 0, len(azimuth_axis)-1)],
                elevation_axis[np.clip(ie, 0, len(elevation_axis)-1)],
            ], axis=-1).astype(np.float32)
            metrics.update(pred_pcs[b], gt_pts)

    result = metrics.compute()
    result["loss"] = total_loss / max(n_batches, 1)
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True,         help="Path to YAML config")
    parser.add_argument("--resume",       default=None,          help="Checkpoint to resume from")
    parser.add_argument("--fast_dev_run", action="store_true",   help="1 batch per epoch (debug)")
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    # Config
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Paper setup uses 8 GPUs. On 1x24GB, RHSS(8 scans) can exceed memory;
    # default to 4 scans unless the user already set RHSS_N_PATTERNS.
    if device.type == "cuda" and "RHSS_N_PATTERNS" not in os.environ:
        os.environ["RHSS_N_PATTERNS"] = "4"
        warnings.warn(
            "RHSS_N_PATTERNS not set: defaulting to 4 for single-GPU memory safety. "
            "Set RHSS_N_PATTERNS=8 to match paper behavior if memory allows.",
            RuntimeWarning,
        )

    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_ds = build_dataset(cfg, "train")
    test_ds  = build_dataset(cfg, "test")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # Model
    model = build_model(cfg).to(device)
    n_params = model.count_parameters()
    print(f"Parametri: {n_params/1e6:.2f}M  (paper: 2.4M)")

    # Optimizer + scheduler (AdamW + cosine, paper Section 4.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    # Focal loss α=0.995, γ=2 (paper Section 4.2)
    criterion = FocalLoss(
        alpha=cfg.training.focal_loss.alpha,
        gamma=cfg.training.focal_loss.gamma,
    )

    # Mixed precision
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Physical axes for metrics (occupancy → point cloud)
    range_axis, az_axis, el_axis = axes_for_dataset(cfg)

    # Resume
    start_epoch = 0
    best_cd     = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cd     = ckpt.get("best_cd", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_cd={best_cd:.4f}")

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{cfg.training.epochs}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Backward-compatible: prefer training.log_every_n_steps, fall back to logging.log_every_n_steps.
        log_every = 50
        if OmegaConf.select(cfg, "training.log_every_n_steps") is not None:
            log_every = int(cfg.training.log_every_n_steps)
        elif OmegaConf.select(cfg, "logging.log_every_n_steps") is not None:
            log_every = int(cfg.logging.log_every_n_steps)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            epoch, log_every=log_every,
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  elapsed={elapsed:.0f}s")

        if args.fast_dev_run:
            print("fast_dev_run: stop dopo 1 epoch.")
            break

        # Periodic evaluation
        if (epoch + 1) % cfg.checkpoint.keep_last_n == 0 or epoch == cfg.training.epochs - 1:
            metrics = evaluate(
                model, test_loader, criterion, device,
                range_axis, az_axis, el_axis,
            )
            print(f"  Test — loss={metrics['loss']:.4f} "
                  f"CD={metrics.get('CD', float('nan')):.4f}  "
                  f"UCD={metrics.get('UCD', float('nan')):.4f}  "
                  f"N_pts={metrics.get('n_pred_points', 0):.0f}")

            # Save the best model by CD
            cd = metrics.get("CD", float("inf"))
            if cd < best_cd:
                best_cd = cd
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "metrics": metrics, "best_cd": best_cd},
                    ckpt_dir / "best.pth",
                )
                print(f"  → Salvato best.pth (CD={best_cd:.4f})")

        # Periodic checkpoint
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch, "best_cd": best_cd},
            ckpt_dir / f"epoch_{epoch+1:03d}.pth",
        )

    print(f"\nTraining completato. Best CD: {best_cd:.4f}")
    print(f"Paper target (RADIal): CD=0.531, UCD=0.429, N_pts=16963")


if __name__ == "__main__":
    main()
