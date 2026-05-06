"""
Step B2 — Coordinate Transformation (Cartesian → Polar)
======================================================
Transforms the LiDAR point cloud from Cartesian coordinates (x, y, z) into polar
coordinates (range, azimuth, elevation) compatible with the radar tensor grid.

REF: Paper Section 3.2, Step (2)
     "The polar coordinate system (range, azimuth, elevation) better
      captures radar feature distributions across distances. We transform
      LiDAR point clouds from Cartesian to polar coordinates."

Adopted convention (must match the DoA estimation convention):
    range     r   = sqrt(x² + y² + z²)         [meters]
    azimuth   az  = atan2(y, x)                [radians, optionally converted to degrees]
    elevation el  = atan2(z, sqrt(x² + y²))    [radians, optionally converted to degrees]

Note: azimuth/elevation conventions depend on the radar coordinate frame.
Verify with your hardware specs and the dataset calibration notes.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Conversion utilities (fully specified math, no TODOs required)
# ------------------------------------------------------------------

def cartesian_to_spherical_rad(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates in RADIANS.
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
    Convert Cartesian coordinates to 3D polar coordinates.

    Args:
        points_xyz: (N, 3) — columns [x, y, z].
        degrees: if True, azimuth/elevation are returned in degrees; otherwise in radians.

    Returns:
        points_polar: (N, 3) — columns [range, azimuth, elevation].
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
    Convert polar coordinates to Cartesian coordinates.

    Args:
        points_polar: (N, 3) — columns [range, azimuth, elevation].
        degrees: if True, azimuth/elevation are provided in degrees.

    Returns:
        points_xyz: (N, 3) — columns [x, y, z].
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
# Binning: continuous polar coords → discrete grid indices
# ------------------------------------------------------------------

def polar_to_grid_indices(
    points_polar: np.ndarray,
    grid_cfg: dict,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map polar points to discrete grid indices [R, A, E].

    Args:
        points_polar: (N, 3) — [range_m, azimuth_deg, elevation_deg].
        grid_cfg: grid parameter dict, e.g.:
                  {
                      "R": 480, "A": 736, "E": 11,
                      "range_min_m": 0.0,   "range_max_m": 50.0,
                      "azimuth_min_deg": -75.0, "azimuth_max_deg": 75.0,
                      "elevation_min_deg": -4.0, "elevation_max_deg": 6.0,
                  }
        clip: if True, clip indices to grid bounds.

    Returns:
        indices: (M, 3) — integer indices [idx_R, idx_A, idx_E].
        valid_mask: (N,) bool — True for points inside the physical grid bounds.
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

    # Normalize to [0, 1] then scale to grid resolution
    idx_r  = ((r  - r_min)  / (r_max  - r_min))  * R
    idx_az = ((az - az_min) / (az_max - az_min)) * A
    idx_el = ((el - el_min) / (el_max - el_min)) * E

    # Valid mask (inside the physical bounds)
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
# Full pipeline: Cartesian LiDAR → radar grid indices
# ------------------------------------------------------------------

class LiDARPolarTransformer:
    """
    Transform a LiDAR point cloud (already in radar frame after calibration)
    from Cartesian coordinates to polar grid indices compatible with the radar grid.

    Typical usage:
        transformer = LiDARPolarTransformer(grid_cfg)
        pts_polar   = transformer.to_polar(pts_xyz_in_radar_frame)
        indices, valid = transformer.to_grid_indices(pts_polar)
    """

    def __init__(self, grid_cfg: dict) -> None:
        self.grid_cfg = grid_cfg

    def to_polar(self, points_xyz: np.ndarray, degrees: bool = True) -> np.ndarray:
        """Convert (N, 3) Cartesian to (N, 3) polar [r, az, el]."""
        return cartesian_to_polar(points_xyz, degrees=degrees)

    def to_grid_indices(
        self, points_polar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map (N, 3) polar to (M, 3) grid indices and a valid mask."""
        return polar_to_grid_indices(points_polar, self.grid_cfg)

    def transform_full(
        self, points_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline: Cartesian → polar → grid indices.

        Returns:
            pts_polar: (N, 3) — [range, az, el] in degrees
            indices:   (M, 3) — indices [idx_R, idx_A, idx_E]
            valid:     (N,)   — bool mask of valid points
        """
        pts_polar = self.to_polar(points_xyz)
        indices, valid = self.to_grid_indices(pts_polar)
        return pts_polar, indices, valid
