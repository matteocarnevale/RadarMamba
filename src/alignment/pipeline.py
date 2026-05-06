"""
Cross-Modal Alignment Pipeline — 5-step orchestrator
===================================================
Runs the 5 steps described in Paper Section 3.2 / Fig. 2 (blue path) to produce:
    - lidar_occupancy: np.ndarray (R, A, E) float32  → ground truth
    - fused_cloud:     np.ndarray (N, 3) float32     → fused LiDAR+radar cloud (debug)

Typical usage:
    pipeline = AlignmentPipeline.from_config(cfg)
    result   = pipeline.run(lidar_pts, radar_pts_xyz, calibration)

    occ = result["lidar_occupancy"]   # → GT input for training
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.alignment.ground_removal import GroundRemover
from src.alignment.coordinate_transform import LiDARPolarTransformer
from src.alignment.radar_lidar_filter import RadarLiDARFilter
from src.alignment.fov_alignment import FoVAligner
from src.alignment.voxelization import Voxelizer
from src.calibration.sensor_calibration import SensorCalibration


class AlignmentPipeline:
    """
    Cross-modal alignment orchestrator (Fig. 2, blue path).

    Steps:
        B1  GroundRemover        — ground removal on LiDAR (Patchwork++)
        B2  LiDARPolarTransformer — Cartesian → polar (after calibration)
        B3  RadarLiDARFilter     — KD-Tree filtering + radar–LiDAR fusion
        B4  FoVAligner           — crop to the common field of view
        B5  Voxelizer            — occupancy grid [R, A, E]
    """

    def __init__(
        self,
        ground_remover: GroundRemover,
        polar_transformer: LiDARPolarTransformer,
        radar_lidar_filter: RadarLiDARFilter,
        fov_aligner: FoVAligner,
        voxelizer: Voxelizer,
    ) -> None:
        self.ground_remover = ground_remover
        self.polar_transformer = polar_transformer
        self.radar_lidar_filter = radar_lidar_filter
        self.fov_aligner = fov_aligner
        self.voxelizer = voxelizer

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "AlignmentPipeline":
        """
        Build the pipeline from YAML config parameters.

        Args:
            cfg: OmegaConf DictConfig loaded from configs/radial.yaml
                 or configs/radelft.yaml.
        """
        pp_cfg   = cfg.preprocessing
        grid_cfg = dict(cfg.dataset.grid)

        return cls(
            ground_remover     = GroundRemover(dict(pp_cfg.ground_removal)),
            polar_transformer  = LiDARPolarTransformer(grid_cfg),
            radar_lidar_filter = RadarLiDARFilter(
                distance_threshold_m=pp_cfg.kdtree_distance_threshold_m
            ),
            fov_aligner = FoVAligner.from_config(grid_cfg),
            voxelizer   = Voxelizer(grid_cfg),
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        lidar_pts_xyz: np.ndarray,
        radar_pts_xyz: np.ndarray,
        calibration: SensorCalibration,
        return_debug: bool = False,
    ) -> dict:
        """
        Run the 5 alignment steps on a single frame.

        Args:
            lidar_pts_xyz: (N_l, 3+) — LiDAR in LiDAR frame (Cartesian x, y, z in meters).
            radar_pts_xyz: (N_r, 3+) — radar 4D points already extracted by DoA, in radar frame,
                           columns [x, y, z, ...]. These should be pre-filtered with a low CFAR
                           threshold (not the hard CFAR from the gray path in Fig. 2).
            calibration:   SensorCalibration — radar↔LiDAR transform.
            return_debug:  if True, includes intermediate clouds in the output.

        Returns:
            Dict with keys:
                "lidar_occupancy":  np.ndarray (R, A, E) float32  — ground truth
                "fused_cloud":      np.ndarray (M, 3)             — fused cloud (optional)
                "lidar_polar":      np.ndarray (N, 3)             — LiDAR in polar coords (optional)
        """

        # ── Step B1: LiDAR ground removal ────────────────────────────
        lidar_no_ground, _ = self.ground_remover.remove_ground(lidar_pts_xyz)

        # ── Step B2: coordinate transform ─────────────────────────────
        # First, bring LiDAR into radar frame using calibration
        lidar_in_radar_frame = calibration.lidar_to_radar(lidar_no_ground[:, :3])

        # Then convert to polar coordinates
        lidar_polar, lidar_grid_idx, lidar_valid = self.polar_transformer.transform_full(
            lidar_in_radar_frame
        )

        # ── Step B3: radar vs LiDAR KD-Tree filter + fusion ───────────
        # Radar points must be in the same frame as LiDAR (radar frame)
        radar_pts_in_radar_frame = radar_pts_xyz[:, :3]  # già in frame radar

        filter_result = self.radar_lidar_filter.run(
            radar_pts_in_radar_frame,
            lidar_in_radar_frame,   # LiDAR already in radar frame (Cartesian)
            fuse=True,
        )
        fused_cloud_xyz = filter_result["fused"]  # (M, 3) Cartesian in radar frame

        # ── Step B4: FoV alignment (on LiDAR polar cloud) ─────────────
        # Crop LiDAR (already in polar coords) to the radar FoV
        lidar_polar_fov, fov_mask = self.fov_aligner.crop_polar(lidar_polar)

        # ── Step B5: voxelization → occupancy grid ────────────────────
        lidar_occupancy = self.voxelizer.from_polar(lidar_polar_fov)

        result: dict = {"lidar_occupancy": lidar_occupancy}

        if return_debug:
            result["fused_cloud"] = fused_cloud_xyz
            result["lidar_polar"] = lidar_polar
            result["lidar_polar_fov"] = lidar_polar_fov
            result["lidar_no_ground"] = lidar_no_ground

        return result
