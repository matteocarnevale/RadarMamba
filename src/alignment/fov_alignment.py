"""
Passo B4 — Field of View Alignment
====================================
Ritaglia sia il point cloud radar che il LiDAR alla stessa copertura
angolare e di range, così le due modalità hanno lo stesso "campo visivo".

REF: Paper Section 3.2, Step (4):
     "Radar generally has a longer sensing range but poorer angular
      resolution than LiDAR. To ensure better feature alignment between
      the two sensors, we crop both point clouds to match the same
      field of view (FoV)."

Il FoV è definito nei config:
    RADIal:  range [0,50]m, azimuth [-75°,75°], elevation [-4°,6°]
    RaDelft: range [0,50]m, azimuth [-70°,70°], elevation [-15°,15°]
"""

from __future__ import annotations

import numpy as np


def crop_fov_polar(
    points_polar: np.ndarray,
    range_min: float,
    range_max: float,
    azimuth_min_deg: float,
    azimuth_max_deg: float,
    elevation_min_deg: float,
    elevation_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritaglia un point cloud in coordinate POLARI al FoV specificato.

    Args:
        points_polar: np.ndarray, shape (N, 3+) — [range_m, azimuth_deg, elevation_deg, ...].
        range_min/max: limiti di range in metri.
        azimuth_min/max_deg: limiti di azimuth in gradi.
        elevation_min/max_deg: limiti di elevation in gradi.

    Returns:
        cropped: np.ndarray, shape (M, K) — punti all'interno del FoV.
        mask:    np.ndarray, shape (N,) bool.
    """
    r  = points_polar[:, 0]
    az = points_polar[:, 1]
    el = points_polar[:, 2]

    mask = (
        (r  >= range_min)       & (r  <= range_max)       &
        (az >= azimuth_min_deg) & (az <= azimuth_max_deg) &
        (el >= elevation_min_deg) & (el <= elevation_max_deg)
    )
    return points_polar[mask], mask


def crop_fov_cartesian(
    points_xyz: np.ndarray,
    range_min: float,
    range_max: float,
    azimuth_min_deg: float,
    azimuth_max_deg: float,
    elevation_min_deg: float,
    elevation_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritaglia un point cloud in coordinate CARTESIANE al FoV specificato.
    Converte internamente in polari per verificare i limiti angolari.

    Args:
        points_xyz: np.ndarray, shape (N, 3+) — [x, y, z, ...].
        (altri parametri come crop_fov_polar)

    Returns:
        cropped: np.ndarray, shape (M, K).
        mask:    np.ndarray, shape (N,) bool.
    """
    x = points_xyz[:, 0].astype(np.float64)
    y = points_xyz[:, 1].astype(np.float64)
    z = points_xyz[:, 2].astype(np.float64)

    r  = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

    mask = (
        (r  >= range_min)         & (r  <= range_max)         &
        (az >= azimuth_min_deg)   & (az <= azimuth_max_deg)   &
        (el >= elevation_min_deg) & (el <= elevation_max_deg)
    )
    return points_xyz[mask], mask


class FoVAligner:
    """
    Applica il ritaglio FoV a radar e LiDAR in modo coerente.

    Uso tipico:
        aligner = FoVAligner.from_config(grid_cfg)
        radar_cropped, _ = aligner.crop_radar_polar(radar_polar_pts)
        lidar_cropped, _ = aligner.crop_lidar_polar(lidar_polar_pts)
    """

    def __init__(
        self,
        range_min: float,
        range_max: float,
        azimuth_min_deg: float,
        azimuth_max_deg: float,
        elevation_min_deg: float,
        elevation_max_deg: float,
    ) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.azimuth_min = azimuth_min_deg
        self.azimuth_max = azimuth_max_deg
        self.elevation_min = elevation_min_deg
        self.elevation_max = elevation_max_deg

    @classmethod
    def from_config(cls, grid_cfg: dict) -> "FoVAligner":
        """Crea da dizionario di configurazione (sezione grid: del YAML)."""
        return cls(
            range_min=grid_cfg["range_min_m"],
            range_max=grid_cfg["range_max_m"],
            azimuth_min_deg=grid_cfg["azimuth_min_deg"],
            azimuth_max_deg=grid_cfg["azimuth_max_deg"],
            elevation_min_deg=grid_cfg["elevation_min_deg"],
            elevation_max_deg=grid_cfg["elevation_max_deg"],
        )

    def crop_polar(self, points_polar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ritaglia punti in coordinate polari."""
        return crop_fov_polar(
            points_polar,
            self.range_min, self.range_max,
            self.azimuth_min, self.azimuth_max,
            self.elevation_min, self.elevation_max,
        )

    def crop_cartesian(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ritaglia punti in coordinate cartesiane."""
        return crop_fov_cartesian(
            points_xyz,
            self.range_min, self.range_max,
            self.azimuth_min, self.azimuth_max,
            self.elevation_min, self.elevation_max,
        )
