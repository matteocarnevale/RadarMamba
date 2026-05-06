"""
Step B1 — Ground Point Removal (Patchwork++)
============================================
REF: Paper Section 3.2, Step (1)
REF: RaDelft data_preparation.py::remove_ground_points_patchwork

pypatchworkpp API (exact, from RaDelft repo):
    params = pypatchworkpp.Parameters()
    pw = pypatchworkpp.patchworkpp(params)
    pw.estimateGround(point_cloud)          # point_cloud: (N, 3+) float32/64
    nonground = pw.getNonground()           # (M, 3) ndarray
    ground    = pw.getGround()             # (G, 3) ndarray
    nonground_idx = pw.getNongroundIndices() # (M,) int
    ground_idx    = pw.getGroundIndices()    # (G,) int

Install Patchwork++:
    pip install pypatchworkpp
    or: https://github.com/url-kaist/patchwork-plusplus
"""

from __future__ import annotations

import numpy as np

try:
    import pypatchworkpp
    PATCHWORKPP_AVAILABLE = True
except ImportError:
    PATCHWORKPP_AVAILABLE = False


class GroundRemover:
    """
    Remove ground points from LiDAR point clouds via Patchwork++.
    Replicates exactly what RaDelft's `prepare_lidar_pointcloud` does.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """
        Args:
            cfg: Optional parameter dictionary (overrides Patchwork++ defaults).
                 If None, uses Patchwork++ defaults (good out-of-the-box).
                 Recognized keys: sensor_height, num_iter, num_lpr,
                 num_min_pts, th_seeds, th_dist.
        """
        self.cfg = cfg or {}
        if not PATCHWORKPP_AVAILABLE:
            import warnings
            warnings.warn(
                "pypatchworkpp is not available — falling back to a simple z-height rule. "
                "Install with: pip install pypatchworkpp  "
                "or from: https://github.com/url-kaist/patchwork-plusplus",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Patchwork++ (exactly as in RaDelft data_preparation.py)
    # ------------------------------------------------------------------

    def _patchwork_remove(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Call Patchwork++ exactly as in RaDelft::remove_ground_points_patchwork.

        Args:
            xyz: (N, 3+) float32 — point cloud with at least x, y, z.

        Returns:
            nonground: (M, 3) — non-ground points.
            ground:    (G, 3) — ground points.
        """
        params = pypatchworkpp.Parameters()

        # Override parameters if provided
        if "sensor_height" in self.cfg:
            params.sensor_height = float(self.cfg["sensor_height"])
        if "num_iter" in self.cfg:
            params.num_iter = int(self.cfg["num_iter"])
        if "num_lpr" in self.cfg:
            params.num_lpr = int(self.cfg["num_lpr"])
        if "num_min_pts" in self.cfg:
            params.num_min_pts = int(self.cfg["num_min_pts"])
        if "th_seeds" in self.cfg:
            params.th_seeds = float(self.cfg["th_seeds"])
        if "th_dist" in self.cfg:
            params.th_dist = float(self.cfg["th_dist"])

        pw = pypatchworkpp.patchworkpp(params)

        # Patchwork++ expects float32/float64 with shape (N, 3+)
        xyz_input = xyz[:, :3].astype(np.float32)
        pw.estimateGround(xyz_input)

        nonground = pw.getNonground()   # (M, 3)
        ground    = pw.getGround()      # (G, 3)
        return nonground, ground

    # ------------------------------------------------------------------
    # z-height fallback (only for development without pypatchworkpp)
    # ------------------------------------------------------------------

    def _height_fallback(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple z-height threshold fallback.
        RaDelft additionally filters z > -2 after Patchwork++; we apply only that.
        """
        # Conservative threshold: ground below -1.5 m from sensor origin
        z_thresh = self.cfg.get("fallback_z_thresh", -1.5)
        non_ground_mask = xyz[:, 2] > z_thresh
        return xyz[non_ground_mask], xyz[~non_ground_mask]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove_ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove ground points.

        Replicates exactly RaDelft::prepare_lidar_pointcloud:
            1. remove_ground_points_patchwork (Patchwork++)
            2. Keep only xyz (cols 0:3)
            3. Filter points with z > -2  (`cleaning_ego_car` is separate)

        Args:
            points: (N, 3+) — [x, y, z, ...] in Cartesian coordinates.

        Returns:
            non_ground: (M, K) — non-ground points (K = original column count).
            ground:     (G, K) — ground points.
        """
        if len(points) == 0:
            return points.copy(), points[:0].copy()

        if PATCHWORKPP_AVAILABLE:
            non_ground_xyz, ground_xyz = self._patchwork_remove(points)
        else:
            non_ground_xyz, ground_xyz = self._height_fallback(points)

        # Apply z > -2 filter as in RaDelft (removes points below the road surface)
        # REF: data_preparation.py line: lidar_point_cloud = lidar_point_cloud[lidar_point_cloud[:, 2] > -2]
        non_ground_xyz = non_ground_xyz[non_ground_xyz[:, 2] > -2.0]

        return non_ground_xyz, ground_xyz

    def remove_ground_with_ego(
        self, points: np.ndarray, ego_x_max: float = 1.0, ego_y_max: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ground removal + ego-vehicle point removal.

        REF: RaDelft::cleaning_ego_car
            Remove points with x<1 AND |y|<1 (vehicle roof points).

        Args:
            points: (N, 3+).
            ego_x_max: x-threshold for ego-car removal (default: 1.0 m).
            ego_y_max: |y|-threshold for ego-car removal (default: 1.0 m).

        Returns:
            clean: (M, K) — points without ground and without ego car.
            ground: (G, K).
        """
        non_ground, ground = self.remove_ground(points)

        if len(non_ground) == 0:
            return non_ground, ground

        # Remove ego-car points: x < ego_x_max AND |y| < ego_y_max
        # REF: cleaning_ego_car in data_preparation.py
        ego_mask = (non_ground[:, 0] < ego_x_max) & (np.abs(non_ground[:, 1]) < ego_y_max)
        clean = non_ground[~ego_mask]
        return clean, ground
