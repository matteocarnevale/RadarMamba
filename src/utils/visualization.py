"""
Visualizzazione di point cloud e occupancy grids
=================================================
Funzioni per debug del preprocessing e dei risultati del modello.
Produce BEV (Bird's Eye View) e view 3D interattivi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------
# BEV (Bird's Eye View) — proiezione sul piano Range-Azimuth
# ------------------------------------------------------------------

def plot_bev_occupancy(
    occupancy: np.ndarray,
    title: str = "BEV Occupancy",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Visualizza la griglia di occupancy in Bird's Eye View (proietta su RA).

    Args:
        occupancy: np.ndarray, shape (R, A, E) o (R, A) — griglia di occupancy.
        title:     titolo del plot.
        save_path: se fornito, salva l'immagine invece di mostrarla.
        figsize:   dimensioni della figura.
    """
    if occupancy.ndim == 3:
        # Proietta su RA: 1 se c'è almeno un voxel occupato lungo E
        bev = (occupancy.max(axis=-1) > 0).astype(np.float32)
    else:
        bev = occupancy.astype(np.float32)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Trasponi: asse x = Azimuth (A), asse y = Range (R)
    im = ax.imshow(bev, origin="lower", aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Azimuth bin →")
    ax.set_ylabel("← Range bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Occupancy")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def plot_bev_comparison(
    lidar_occ:  np.ndarray,
    radar_pred: np.ndarray,
    title:      str = "BEV: LiDAR (bianco) vs Radar Enhanced (rosso)",
    save_path:  Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Confronto BEV tra occupancy LiDAR (GT) e predizione radar.
    Riproduce la Fig. 3 del paper.

    Args:
        lidar_occ:  (R, A, E) — ground truth LiDAR.
        radar_pred: (R, A, E) — predizione modello (dopo sigmoid > threshold).
        title, save_path, figsize: come plot_bev_occupancy.
    """
    # Proietta su RA
    lidar_bev  = (lidar_occ.max(axis=-1) > 0)
    radar_bev  = (radar_pred.max(axis=-1) > 0)

    # Crea immagine RGB
    R, A = lidar_bev.shape
    img = np.zeros((R, A, 3), dtype=np.float32)

    # LiDAR → bianco (1,1,1)
    img[lidar_bev] = [1.0, 1.0, 1.0]
    # Radar enhanced → rosso (1,0,0)
    # Se coincidono: arancione (1, 0.5, 0)
    img[radar_bev, 0] = 1.0
    overlap = lidar_bev & radar_bev
    img[overlap] = [1.0, 0.5, 0.0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Azimuth →")
    ax.set_ylabel("← Range")
    ax.set_title(title)

    patches = [
        mpatches.Patch(color="white",  label="LiDAR GT"),
        mpatches.Patch(color="red",    label="Radar Enhanced"),
        mpatches.Patch(color="orange", label="Overlap"),
    ]
    ax.legend(handles=patches, loc="upper right", facecolor="black", labelcolor="white")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# Radar cube — visualizzazione canali
# ------------------------------------------------------------------

def plot_radar_cube_channels(
    radar_cube: np.ndarray,
    n_cols: int = 3,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Visualizza i 6 canali del tensore radar (proiezione su RA).

    Args:
        radar_cube: np.ndarray, shape (R, A, E, 6) o (6, R, A, E).
        n_cols:     colonne nel subplot.
        save_path:  path per il salvataggio.
    """
    if radar_cube.ndim == 4 and radar_cube.shape[0] == 6:
        cube = radar_cube.transpose(1, 2, 3, 0)  # (R,A,E,6)
    else:
        cube = radar_cube   # assumiamo (R,A,E,6)

    channel_names = [
        "Intensità t", "Doppler t",
        "Intensità t-1", "Doppler t-1",
        "Intensità t-2", "Doppler t-2",
    ]

    n_channels = cube.shape[-1]
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_channels == 1 else list(axes)

    for i in range(n_channels):
        bev = cube[:, :, :, i].max(axis=-1)   # max su elevation
        axes[i].imshow(bev, origin="lower", aspect="auto", cmap="viridis")
        axes[i].set_title(channel_names[i] if i < len(channel_names) else f"ch {i}")
        axes[i].axis("off")

    for i in range(n_channels, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Radar Cube Channels (proiezione BEV: max su E)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Statistiche del preprocessing
# ------------------------------------------------------------------

def print_preprocessing_stats(
    lidar_occ: np.ndarray,
    radar_cube: np.ndarray,
    rad_map: np.ndarray,
) -> None:
    """Stampa statistiche di un campione preprocessato (utile per debug)."""
    print("=" * 50)
    print("PREPROCESSING STATS")
    print("=" * 50)
    print(f"Occupancy GT:   shape={lidar_occ.shape}, dtype={lidar_occ.dtype}")
    print(f"  Voxel totali: {lidar_occ.size:,}")
    print(f"  Voxel occ.:   {lidar_occ.sum():.0f}  ({100*lidar_occ.mean():.3f}%)")
    print()
    print(f"Radar cube:     shape={radar_cube.shape}, dtype={radar_cube.dtype}")
    print(f"  Intensità ch0 — min={radar_cube[..., 0].min():.3f}, "
          f"max={radar_cube[..., 0].max():.3f}, "
          f"mean={radar_cube[..., 0].mean():.3f}")
    print(f"  Doppler   ch1 — min={radar_cube[..., 1].min():.3f}, "
          f"max={radar_cube[..., 1].max():.3f}")
    print()
    if rad_map is not None:
        print(f"RAD map:        shape={rad_map.shape}, dtype={rad_map.dtype}")
        print(f"  min={rad_map.min():.3f}, max={rad_map.max():.3f}, mean={rad_map.mean():.3f}")
    print("=" * 50)
