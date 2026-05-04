"""
Passo B2 — Coordinate Transformation (Cartesian → Polar)
==========================================================
Trasforma il point cloud LiDAR da coordinate cartesiane (x, y, z)
a coordinate polari (range, azimuth, elevation) compatibili con la
griglia del tensore radar.

REF: Paper Section 3.2, Step (2)
     "The polar coordinate system (range, azimuth, elevation) better
      captures radar feature distributions across distances. We transform
      LiDAR point clouds from Cartesian to polar coordinates."

Convenzione adottata (deve essere coerente con la DoA estimation):
    range     r   = sqrt(x² + y² + z²)         [metri]
    azimuth   az  = atan2(y, x)                 [radianti, poi convertito in gradi]
    elevation el  = atan2(z, sqrt(x² + y²))     [radianti, poi convertito in gradi]

Nota: la convezione di azimuth/elevation dipende dall'orientamento del radar.
Verifica con le specifiche hardware e le note di calibrazione dei dataset.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Funzioni di conversione (matematica completa — non richiedono TODO)
# ------------------------------------------------------------------

def cartesian_to_spherical_rad(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Converte coordinate cartesiane in sferiche in RADIANTI.
    Replicates RaDelft::data_preparation.cartesian_to_spherical.

    REF: data_preparation.py::cartesian_to_spherical
         range  = sqrt(x²+y²+z²)
         azimuth = arctan2(y, x)
         elevation = arcsin(z / r)

    Returns:
        (N, 3) — [range_m, az_rad, el_rad]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    r  = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):
        el = np.arcsin(np.clip(z / np.where(r > 0, r, 1e-9), -1.0, 1.0))
    return np.stack([r, az, el], axis=1)


def cartesian_to_polar(
    points_xyz: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    Converte coordinate cartesiane in coordinate polari 3D.

    Args:
        points_xyz: np.ndarray, shape (N, 3) — colonne [x, y, z].
        degrees: se True, azimuth ed elevation sono in gradi; altrimenti radianti.

    Returns:
        points_polar: np.ndarray, shape (N, 3) — colonne [range, azimuth, elevation].
    """
    x = points_xyz[:, 0].astype(np.float64)
    y = points_xyz[:, 1].astype(np.float64)
    z = points_xyz[:, 2].astype(np.float64)

    r  = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)                           # [-π, π]
    # Use arcsin for elevation — consistent with RaDelft data_preparation.py
    # REF: data_preparation.cartesian_to_spherical uses arcsin(z/r)
    with np.errstate(invalid="ignore"):
        el = np.arcsin(np.clip(z / np.where(r > 0, r, 1e-9), -1.0, 1.0))  # [-π/2, π/2]

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    return np.stack([r, az, el], axis=-1)


def polar_to_cartesian(
    points_polar: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    Converte coordinate polari in coordinate cartesiane.

    Args:
        points_polar: np.ndarray, shape (N, 3) — colonne [range, azimuth, elevation].
        degrees: se True, azimuth ed elevation sono in gradi.

    Returns:
        points_xyz: np.ndarray, shape (N, 3) — colonne [x, y, z].
    """
    r  = points_polar[:, 0].astype(np.float64)
    az = points_polar[:, 1].astype(np.float64)
    el = points_polar[:, 2].astype(np.float64)

    if degrees:
        az = np.radians(az)
        el = np.radians(el)

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    return np.stack([x, y, z], axis=-1)


# ------------------------------------------------------------------
# Binning: da polari continui a indici di griglia discreta
# ------------------------------------------------------------------

def polar_to_grid_indices(
    points_polar: np.ndarray,
    grid_cfg: dict,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mappa punti in coordinate polari agli indici discreti della griglia [R, A, E].

    Args:
        points_polar: np.ndarray, shape (N, 3) — [range_m, azimuth_deg, elevation_deg].
        grid_cfg: dizionario con i parametri della griglia, es.:
                  {
                      "R": 480, "A": 736, "E": 11,
                      "range_min_m": 0.0,   "range_max_m": 50.0,
                      "azimuth_min_deg": -75.0, "azimuth_max_deg": 75.0,
                      "elevation_min_deg": -4.0, "elevation_max_deg": 6.0,
                  }
        clip: se True, taglia punti fuori dai limiti della griglia.

    Returns:
        indices: np.ndarray, shape (M, 3) — [idx_R, idx_A, idx_E] interi.
        valid_mask: np.ndarray, shape (N,) bool — True per i punti dentro la griglia.
    """
    R = grid_cfg["R"]
    A = grid_cfg["A"]
    E = grid_cfg["E"]

    r_min, r_max   = grid_cfg["range_min_m"],      grid_cfg["range_max_m"]
    az_min, az_max = grid_cfg["azimuth_min_deg"],   grid_cfg["azimuth_max_deg"]
    el_min, el_max = grid_cfg["elevation_min_deg"], grid_cfg["elevation_max_deg"]

    r  = points_polar[:, 0]
    az = points_polar[:, 1]
    el = points_polar[:, 2]

    # Normalizza [0, 1] → scala sulla dimensione della griglia
    idx_r  = ((r  - r_min)  / (r_max  - r_min))  * R
    idx_az = ((az - az_min) / (az_max - az_min)) * A
    idx_el = ((el - el_min) / (el_max - el_min)) * E

    # Maschera punti validi (dentro i limiti fisici del sensore)
    valid = (
        (r  >= r_min)  & (r  <  r_max)  &
        (az >= az_min) & (az <  az_max) &
        (el >= el_min) & (el <  el_max)
    )

    idx_r  = np.clip(idx_r.astype(np.int32),  0, R - 1)
    idx_az = np.clip(idx_az.astype(np.int32), 0, A - 1)
    idx_el = np.clip(idx_el.astype(np.int32), 0, E - 1)

    indices = np.stack([idx_r, idx_az, idx_el], axis=-1)

    if not clip:
        indices = indices[valid]

    return indices, valid


# ------------------------------------------------------------------
# Pipeline completa LiDAR Cartesiano → indici griglia radar
# ------------------------------------------------------------------

class LiDARPolarTransformer:
    """
    Trasforma un point cloud LiDAR (già in frame radar dopo calibrazione)
    da coordinate cartesiane a indici di griglia polare compatibili col radar.

    Uso tipico:
        transformer = LiDARPolarTransformer(grid_cfg)
        pts_polar   = transformer.to_polar(pts_xyz_in_radar_frame)
        indices, valid = transformer.to_grid_indices(pts_polar)
    """

    def __init__(self, grid_cfg: dict) -> None:
        self.grid_cfg = grid_cfg

    def to_polar(self, points_xyz: np.ndarray, degrees: bool = True) -> np.ndarray:
        """Converte (N,3) cartesiane in (N,3) polari [r, az, el]."""
        return cartesian_to_polar(points_xyz, degrees=degrees)

    def to_grid_indices(
        self, points_polar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mappa (N,3) polari in (M,3) indici griglia e maschera validi."""
        return polar_to_grid_indices(points_polar, self.grid_cfg)

    def transform_full(
        self, points_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline completa: cartesiano → polari → indici griglia.

        Returns:
            pts_polar: (N, 3) — [range, az, el] in gradi
            indices:   (M, 3) — indici [idx_R, idx_A, idx_E]
            valid:     (N,)   — bool mask punti validi
        """
        pts_polar = self.to_polar(points_xyz)
        indices, valid = self.to_grid_indices(pts_polar)
        return pts_polar, indices, valid
