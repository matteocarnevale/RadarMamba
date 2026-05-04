"""
Test unitari per la voxelizzazione.
Verifica che la pipeline coordinate_transform → voxelization produca
output della shape e dei valori corretti.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.alignment.coordinate_transform import (
    cartesian_to_polar,
    polar_to_cartesian,
    polar_to_grid_indices,
)
from src.alignment.voxelization import voxelize_lidar, voxelize_from_polar_points


GRID_CFG = {
    "R": 48, "A": 74, "E": 11,
    "range_min_m": 0.0, "range_max_m": 50.0,
    "azimuth_min_deg": -75.0, "azimuth_max_deg": 75.0,
    "elevation_min_deg": -4.0, "elevation_max_deg": 6.0,
}


class TestCoordinateTransform:

    def test_roundtrip_cartesian_polar(self):
        """Converti cartesiano→polare→cartesiano e verifica la correttezza."""
        pts = np.array([[10.0, 5.0, 1.5], [20.0, -3.0, 0.5], [1.0, 0.0, 0.0]])
        pts_polar = cartesian_to_polar(pts, degrees=True)
        pts_back  = polar_to_cartesian(pts_polar, degrees=True)
        np.testing.assert_allclose(pts, pts_back, atol=1e-6, err_msg="Roundtrip failed")

    def test_polar_range_positive(self):
        """Il range deve essere sempre positivo."""
        pts = np.random.randn(100, 3).astype(np.float64)
        pts_polar = cartesian_to_polar(pts)
        assert (pts_polar[:, 0] >= 0).all(), "Range deve essere >= 0"

    def test_azimuth_range(self):
        """Azimuth in gradi deve essere in [-180, 180]."""
        pts = np.random.randn(100, 3)
        pts_polar = cartesian_to_polar(pts)
        assert (pts_polar[:, 1] >= -180).all()
        assert (pts_polar[:, 1] <= 180).all()

    def test_grid_indices_in_bounds(self):
        """Gli indici di griglia devono essere all'interno dei limiti."""
        pts = np.array([
            [0.104, 0.0, 0.0],   # range ~1m, az=0°, el=0°
            [25.0,  0.0, 0.0],   # range 25m, az=0°, el=0°
        ])
        pts_polar = cartesian_to_polar(pts)
        indices, valid = polar_to_grid_indices(pts_polar, GRID_CFG)
        assert (indices[:, 0] >= 0).all() and (indices[:, 0] < GRID_CFG["R"]).all()
        assert (indices[:, 1] >= 0).all() and (indices[:, 1] < GRID_CFG["A"]).all()
        assert (indices[:, 2] >= 0).all() and (indices[:, 2] < GRID_CFG["E"]).all()

    def test_point_outside_fov_invalid(self):
        """Punti fuori dal FoV devono avere valid=False."""
        pts_outside = np.array([[200.0, 0.0, 0.0]])   # range 200m > max 50m
        pts_polar = cartesian_to_polar(pts_outside)
        _, valid = polar_to_grid_indices(pts_polar, GRID_CFG)
        assert not valid.any(), "Punto fuori range deve essere invalid"


class TestVoxelization:

    def test_output_shape(self):
        """L'occupancy grid deve avere la shape (R, A, E)."""
        indices = np.array([[5, 10, 3], [20, 30, 7]])
        occ = voxelize_lidar(indices, GRID_CFG["R"], GRID_CFG["A"], GRID_CFG["E"])
        assert occ.shape == (GRID_CFG["R"], GRID_CFG["A"], GRID_CFG["E"])

    def test_output_dtype(self):
        """L'occupancy grid deve essere float32."""
        indices = np.array([[0, 0, 0]])
        occ = voxelize_lidar(indices, GRID_CFG["R"], GRID_CFG["A"], GRID_CFG["E"])
        assert occ.dtype == np.float32

    def test_occupied_voxels(self):
        """I voxel corrispondenti agli indici devono essere 1."""
        indices = np.array([[5, 10, 3], [20, 30, 7]])
        occ = voxelize_lidar(indices, GRID_CFG["R"], GRID_CFG["A"], GRID_CFG["E"])
        assert occ[5, 10, 3] == 1.0
        assert occ[20, 30, 7] == 1.0

    def test_binary_values(self):
        """L'occupancy deve contenere solo 0 e 1."""
        pts = np.random.rand(500, 3)
        pts[:, 0] *= 30    # range 0-30m
        pts[:, 1] = pts[:, 1] * 60 - 30   # az -30 to 30 deg
        pts[:, 2] = pts[:, 2] * 5 - 2     # el -2 to 3 deg
        occ = voxelize_from_polar_points(pts, GRID_CFG)
        unique = np.unique(occ)
        assert set(unique).issubset({0.0, 1.0}), f"Valori non binari: {unique}"

    def test_empty_cloud(self):
        """Cloud vuoto → occupancy tutta zero."""
        occ = voxelize_lidar(
            np.zeros((0, 3), dtype=np.int32),
            GRID_CFG["R"], GRID_CFG["A"], GRID_CFG["E"]
        )
        assert occ.sum() == 0.0

    def test_full_fov_points(self):
        """Punti distribuiti nel FoV → occupancy non-zero."""
        # Genera punti densi nel FoV
        N = 2000
        r  = np.random.uniform(5.0, 45.0, N)
        az = np.random.uniform(-70.0, 70.0, N)
        el = np.random.uniform(-3.0, 5.0, N)
        pts_polar = np.stack([r, az, el], axis=-1)
        occ = voxelize_from_polar_points(pts_polar, GRID_CFG)
        assert occ.sum() > 0, "Nessun voxel occupato con punti nel FoV!"
