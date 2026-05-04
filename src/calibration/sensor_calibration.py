"""
Sensor Calibration Loader
=========================
Carica e gestisce la calibrazione extrinsic tra radar e LiDAR.

Entrambi i dataset forniscono file di calibrazione:
- RADIal: parametri extrinsic nel repo (verificare con DBReader)
- RaDelft: calibration.json per ogni scena

La calibrazione è necessaria per il Passo B2 (Coordinate Transform):
trasformare i punti LiDAR in coordinate polari coerenti col radar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SensorCalibration:
    """
    Mantiene le matrici di trasformazione tra i sistemi di riferimento
    del radar e del LiDAR.

    Convenzione:
        T_lidar_to_radar : np.ndarray, shape (4, 4)
            Matrice omogenea che trasforma punti da frame LiDAR a frame Radar.
            p_radar = T_lidar_to_radar @ p_lidar_homogeneous

        T_radar_to_lidar : np.ndarray, shape (4, 4)
            Inversa di T_lidar_to_radar (calcolata automaticamente).
    """

    def __init__(
        self,
        T_lidar_to_radar: np.ndarray,
        radar_origin_offset: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            T_lidar_to_radar: Matrice omogenea 4x4.
            radar_origin_offset: Traslazione aggiuntiva (x, y, z) in metri
                se il radar non coincide esattamente con l'origine.
        """
        assert T_lidar_to_radar.shape == (4, 4), "Expected 4x4 homogeneous matrix"
        self.T_lidar_to_radar = T_lidar_to_radar.astype(np.float64)
        self.T_radar_to_lidar = np.linalg.inv(T_lidar_to_radar)
        self.radar_origin_offset = radar_origin_offset if radar_origin_offset is not None else np.zeros(3)

    # ------------------------------------------------------------------
    # Factory methods — uno per dataset
    # ------------------------------------------------------------------

    @classmethod
    def from_radial(cls, sequence_path: str | Path) -> "SensorCalibration":
        """
        Carica la calibrazione dal formato RADIal.

        REF: https://github.com/valeoai/RADIal
        Il repo RADIal fornisce parametri extrinsic insieme al dataset.
        Usa la libreria DBReader per leggerli, oppure caricali dal file
        di calibrazione che trovi nella cartella della sequenza.

        Args:
            sequence_path: Path alla cartella della sequenza RADIal
                           (es. data/raw/radial/RADIal_sequence_000).

        Returns:
            SensorCalibration instance.

        TODO:
            1. Esplora la struttura della sequenza RADIal.
            2. Individua il file di calibrazione (es. calibration.json
               o parametri nell'header del file log).
            3. Leggi la rotazione R (3x3) e la traslazione t (3,) del LiDAR
               rispetto al radar (o viceversa — controlla la convenzione).
            4. Costruisci T_lidar_to_radar:
                  T = np.eye(4)
                  T[:3, :3] = R
                  T[:3, 3]  = t
            5. Restituisci cls(T).
        """
        raise NotImplementedError(
            "TODO: implementa il caricamento della calibrazione RADIal.\n"
            "Hint: usa DBReader o leggi il file di calibrazione dalla sequenza."
        )

    @classmethod
    def from_radelft(cls, scene_path: str | Path) -> "SensorCalibration":
        """
        Carica la calibrazione dal formato RaDelft.

        REF: https://github.com/RaDelft/RaDelft-Dataset
        RaDelft fornisce un file calibration.json per ogni scena con
        le trasformazioni radar→lidar e camera→lidar.

        Args:
            scene_path: Path alla cartella della scena RaDelft
                        (es. data/raw/radelft/scene_1).

        Returns:
            SensorCalibration instance.

        TODO:
            1. Carica calibration.json da scene_path.
            2. Estrai la chiave che descrive T_radar_to_lidar (o inversa).
               Nella notazione del repo RaDelft la chiave è probabilmente
               "radar_to_lidar" o "lidar_to_radar" — controlla il JSON.
            3. Converti in np.ndarray 4x4.
            4. Se il JSON fornisce T_radar_to_lidar, invertila per ottenere
               T_lidar_to_radar.
            5. Restituisci cls(T_lidar_to_radar).
        """
        raise NotImplementedError(
            "TODO: implementa il caricamento della calibrazione RaDelft.\n"
            "Hint: carica scene_path/calibration.json e leggi la chiave corretta."
        )

    @classmethod
    def identity(cls) -> "SensorCalibration":
        """Calibrazione identità — utile per test o dati pre-allineati."""
        return cls(np.eye(4))

    # ------------------------------------------------------------------
    # Trasformazione punti
    # ------------------------------------------------------------------

    def lidar_to_radar(self, points_lidar: np.ndarray) -> np.ndarray:
        """
        Trasforma un point cloud da sistema LiDAR a sistema Radar.

        Args:
            points_lidar: np.ndarray, shape (N, 3) — coordinate x,y,z in frame LiDAR.

        Returns:
            points_radar: np.ndarray, shape (N, 3) — coordinate x,y,z in frame Radar.
        """
        N = points_lidar.shape[0]
        # Aggiungi colonna di uni per coordinate omogenee
        ones = np.ones((N, 1), dtype=points_lidar.dtype)
        pts_h = np.hstack([points_lidar, ones])            # (N, 4)
        pts_radar_h = (self.T_lidar_to_radar @ pts_h.T).T  # (N, 4)
        return pts_radar_h[:, :3]

    def radar_to_lidar(self, points_radar: np.ndarray) -> np.ndarray:
        """
        Trasforma un point cloud da sistema Radar a sistema LiDAR.

        Args:
            points_radar: np.ndarray, shape (N, 3).

        Returns:
            points_lidar: np.ndarray, shape (N, 3).
        """
        N = points_radar.shape[0]
        ones = np.ones((N, 1), dtype=points_radar.dtype)
        pts_h = np.hstack([points_radar, ones])
        pts_lidar_h = (self.T_radar_to_lidar @ pts_h.T).T
        return pts_lidar_h[:, :3]
