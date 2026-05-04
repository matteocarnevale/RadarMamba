"""
Passo B3 — Radar Point Cloud Filtering & LiDAR Fusion
=======================================================
Filtra i punti radar rumorosi usando un KD-Tree costruito sul LiDAR,
poi fonde il radar filtrato con il point cloud LiDAR per creare una
nuvola di punti più densa.

REF: Paper Section 3.2, Step (3) + Eq. (1) e (2):

    p_LiDAR_j = argmin_{p_LiDAR_k ∈ KDTree} || p_Radar_i - p_LiDAR_k ||_2   (Eq. 1)
    d_ij       = || p_Radar_i - p_LiDAR_j ||_2                                (Eq. 2)

    Se d_ij > 0.5 m → punto radar isolato → rimuovi.

Nota sul flusso Fig. 2 (blue path):
    Prima di costruire il KD-Tree, viene applicato un CFAR a bassa soglia
    per rimuovere il rumore grossolano pur mantenendo i punti di qualità.
    Il KD-Tree è poi costruito sul LiDAR (già senza suolo).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


class RadarLiDARFilter:
    """
    Filtra punti radar contro il LiDAR via KD-Tree e fonde i cloud.

    Workflow:
        1. Costruisci KD-Tree dai punti LiDAR (già trasformati in frame radar).
        2. Per ogni punto radar trova il vicino LiDAR più prossimo.
        3. Scarta i punti radar con distanza > distance_threshold.
        4. (Opzionale) Fondi il radar filtrato con il cloud LiDAR.
    """

    def __init__(self, distance_threshold_m: float = 0.5) -> None:
        """
        Args:
            distance_threshold_m: soglia distanza Euclidea (paper: 0.5 m).
                                  Punti radar oltre questa soglia vengono rimossi.
        """
        self.distance_threshold = distance_threshold_m

    def filter_radar_points(
        self,
        radar_points: np.ndarray,
        lidar_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filtra i punti radar mantenendo solo quelli vicini al LiDAR.

        Args:
            radar_points: np.ndarray, shape (N_r, 3) — punti radar in coordinate
                          CARTESIANE nel frame radar (dopo DoA, pre-voxelization).
                          Devono essere nello stesso sistema di riferimento dei punti
                          LiDAR (applica calibrazione prima di chiamare questo metodo).
            lidar_points: np.ndarray, shape (N_l, 3) — punti LiDAR senza suolo,
                          già trasformati nel frame radar (passo B2 li porta in polari,
                          ma qui usiamo ancora le cartesiane per la distanza Euclidea).

        Returns:
            filtered_radar: np.ndarray, shape (M, 3) — punti radar che hanno un
                            vicino LiDAR a distanza <= threshold. M <= N_r.
            keep_mask: np.ndarray, shape (N_r,) bool — True = punto mantenuto.

        Note:
            Le coordinate devono essere nello stesso sistema (es. frame radar
            cartesiano) per calcolare distanze Euclidee 3D corrette.
        """
        if len(radar_points) == 0:
            return radar_points, np.array([], dtype=bool)
        if len(lidar_points) == 0:
            # Nessun punto LiDAR → rimuovi tutto il radar (nessun riferimento)
            return np.empty((0, radar_points.shape[1])), np.zeros(len(radar_points), dtype=bool)

        # Passo 1: costruisci KD-Tree sui punti LiDAR (Eq. 1)
        # TODO: il KD-Tree va costruito solo una volta per frame (non per ogni punto)
        #       Qui lo costruiamo una volta per chiamata — OK per ora.
        kdtree = KDTree(lidar_points[:, :3])

        # Passo 2: per ogni punto radar, trova il vicino LiDAR più prossimo (Eq. 1-2)
        distances, _ = kdtree.query(radar_points[:, :3], k=1, workers=-1)

        # Passo 3: mantieni solo i punti con distanza <= threshold (Eq. 2)
        keep_mask = distances <= self.distance_threshold
        filtered_radar = radar_points[keep_mask]

        return filtered_radar, keep_mask

    def fuse_radar_lidar(
        self,
        filtered_radar: np.ndarray,
        lidar_points: np.ndarray,
    ) -> np.ndarray:
        """
        Fondi il radar filtrato con il point cloud LiDAR.

        REF: Paper Section 3.2, Step (3):
             "The filtered radar point cloud is then aligned and fused
              with the LiDAR cloud."

        Il risultato è una nuvola di punti più densa che combina la
        penetrazione del radar con l'accuratezza geometrica del LiDAR.

        Args:
            filtered_radar: np.ndarray, shape (M, 3+) — punti radar filtrati.
            lidar_points:   np.ndarray, shape (N, 3+) — punti LiDAR.

        Returns:
            fused: np.ndarray, shape (M+N, 3) — cloud fuso (solo colonne x,y,z).

        TODO:
            Considera se mantenere anche i canali extra (intensità, Doppler ecc.)
            o se servono solo le colonne xyz per la voxelizzazione successiva.
            La paper non specifica — per la voxelization bastano le xyz.
        """
        r_xyz = filtered_radar[:, :3]
        l_xyz = lidar_points[:, :3]
        fused = np.vstack([r_xyz, l_xyz])
        return fused

    def run(
        self,
        radar_points: np.ndarray,
        lidar_points: np.ndarray,
        fuse: bool = True,
    ) -> dict:
        """
        Esegue filtro + fusione in un'unica chiamata.

        Args:
            radar_points: (N_r, 3+) — punti radar cartesiani nel frame radar.
            lidar_points: (N_l, 3+) — punti LiDAR cartesiani nel frame radar.
            fuse: se True, concatena radar filtrato + LiDAR.

        Returns:
            dict con chiavi:
                "filtered_radar": punti radar dopo filtro KD-Tree
                "keep_mask":      maschera bool sui punti radar originali
                "fused":          cloud fuso (se fuse=True, altrimenti None)
        """
        filtered_radar, keep_mask = self.filter_radar_points(radar_points, lidar_points)
        fused = self.fuse_radar_lidar(filtered_radar, lidar_points) if fuse else None

        return {
            "filtered_radar": filtered_radar,
            "keep_mask": keep_mask,
            "fused": fused,
        }
