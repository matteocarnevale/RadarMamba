"""
Cross-Modal Alignment Pipeline — orchestratore dei 5 passi
============================================================
Esegue in sequenza i 5 passi descritti in Paper Section 3.2 / Fig. 2
(blue path) per produrre:
    - lidar_occupancy: np.ndarray (R, A, E) float32  → ground truth
    - fused_cloud:     np.ndarray (N, 3) float32     → cloud LiDAR+radar fuso (debug)

Uso tipico:
    pipeline = AlignmentPipeline.from_config(cfg)
    result   = pipeline.run(lidar_pts, radar_pts_xyz, calibration)

    occ = result["lidar_occupancy"]   # → input GT per il training
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
    Orchestratore della cross-modal alignment (Fig. 2, blue path).

    Passi:
        B1  GroundRemover     — rimozione suolo da LiDAR (Patchwork++)
        B2  LiDARPolarTransformer — cartesiano → polare (dopo calibrazione)
        B3  RadarLiDARFilter  — KD-Tree + fusione radar-LiDAR
        B4  FoVAligner        — ritaglio al campo visivo comune
        B5  Voxelizer         — griglia occupancy [R,A,E]
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
        Costruisce la pipeline dai parametri del YAML.

        Args:
            cfg: OmegaConf DictConfig caricato da configs/radial.yaml
                 o configs/radelft.yaml.
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
    # Pipeline principale
    # ------------------------------------------------------------------

    def run(
        self,
        lidar_pts_xyz: np.ndarray,
        radar_pts_xyz: np.ndarray,
        calibration: SensorCalibration,
        return_debug: bool = False,
    ) -> dict:
        """
        Esegue i 5 passi di allineamento su un singolo frame.

        Args:
            lidar_pts_xyz: np.ndarray, shape (N_l, 3+) — LiDAR in frame LiDAR
                           (coordinate cartesiane x, y, z in metri).
            radar_pts_xyz: np.ndarray, shape (N_r, 3+) — punti radar 4D già
                           estratti da DoA, in frame radar, colonne [x, y, z, ...].
                           Questi sono i punti radar pre-filtrati con CFAR basso
                           (non il CFAR duro del percorso grigio di Fig. 2).
            calibration:   SensorCalibration — trasformazione radar↔lidar.
            return_debug:  se True, include cloud intermedi nel risultato.

        Returns:
            dict con chiavi:
                "lidar_occupancy":  np.ndarray (R, A, E) float32  — ground truth
                "fused_cloud":      np.ndarray (M, 3)              — cloud fuso (opt.)
                "lidar_polar":      np.ndarray (N, 3)              — LiDAR in polari (opt.)
        """

        # ── Passo B1: rimozione suolo dal LiDAR ──────────────────────
        lidar_no_ground, _ = self.ground_remover.remove_ground(lidar_pts_xyz)

        # ── Passo B2: trasformazione coordinate ──────────────────────
        # Prima porta LiDAR nel frame radar con la calibrazione
        lidar_in_radar_frame = calibration.lidar_to_radar(lidar_no_ground[:, :3])

        # Poi converte in coordinate polari
        lidar_polar, lidar_grid_idx, lidar_valid = self.polar_transformer.transform_full(
            lidar_in_radar_frame
        )

        # ── Passo B3: filtro KD-Tree radar vs LiDAR + fusione ────────
        # I punti radar devono essere nello stesso frame del LiDAR (frame radar)
        radar_pts_in_radar_frame = radar_pts_xyz[:, :3]  # già in frame radar

        filter_result = self.radar_lidar_filter.run(
            radar_pts_in_radar_frame,
            lidar_in_radar_frame,   # LiDAR già in frame radar (cartesiano)
            fuse=True,
        )
        fused_cloud_xyz = filter_result["fused"]  # (M, 3) cartesiane nel frame radar

        # ── Passo B4: allineamento FoV (su cloud fuso) ───────────────
        # Qui allineamo il cloud LiDAR (già in polari) al FoV del radar
        lidar_polar_fov, fov_mask = self.fov_aligner.crop_polar(lidar_polar)

        # ── Passo B5: voxelizzazione → occupancy grid ─────────────────
        lidar_occupancy = self.voxelizer.from_polar(lidar_polar_fov)

        result: dict = {"lidar_occupancy": lidar_occupancy}

        if return_debug:
            result["fused_cloud"] = fused_cloud_xyz
            result["lidar_polar"] = lidar_polar
            result["lidar_polar_fov"] = lidar_polar_fov
            result["lidar_no_ground"] = lidar_no_ground

        return result
