"""
Passo B1 — Ground Point Removal (Patchwork++)
===============================================
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
    Rimozione del suolo da punto cloud LiDAR via Patchwork++.
    Replicates exactly what RaDelft's prepare_lidar_pointcloud does.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """
        Args:
            cfg: dizionario opzionale di parametri (override Patchwork++ defaults).
                 Se None, usa i default di Patchwork++ (funzionano bene out-of-box).
                 Chiavi riconosciute: sensor_height, num_iter, num_lpr,
                 num_min_pts, th_seeds, th_dist.
        """
        self.cfg = cfg or {}
        if not PATCHWORKPP_AVAILABLE:
            import warnings
            warnings.warn(
                "pypatchworkpp non disponibile — uso fallback sull'altezza z. "
                "Installa con: pip install pypatchworkpp  "
                "o da: https://github.com/url-kaist/patchwork-plusplus",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Patchwork++ (esatto come in RaDelft data_preparation.py)
    # ------------------------------------------------------------------

    def _patchwork_remove(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Chiama Patchwork++ esattamente come in RaDelft::remove_ground_points_patchwork.

        Args:
            xyz: (N, 3+) float32 — punto cloud con almeno x,y,z.

        Returns:
            nonground: (M, 3) — punti senza suolo.
            ground:    (G, 3) — punti del suolo.
        """
        params = pypatchworkpp.Parameters()

        # Override dei parametri se forniti nel cfg
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

        # Patchwork++ vuole float32 o float64 con shape (N, 3+)
        xyz_input = xyz[:, :3].astype(np.float32)
        pw.estimateGround(xyz_input)

        nonground = pw.getNonground()   # (M, 3)
        ground    = pw.getGround()      # (G, 3)
        return nonground, ground

    # ------------------------------------------------------------------
    # Fallback sull'altezza z (solo per sviluppo senza pypatchworkpp)
    # ------------------------------------------------------------------

    def _height_fallback(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Semplice soglia sull'altezza z come fallback.
        RaDelft inoltre filtra z > -2 dopo Patchwork++; applichiamo solo quello.
        """
        # Soglia conservativa: suolo sotto -1.5 m dall'origine del sensore
        z_thresh = self.cfg.get("fallback_z_thresh", -1.5)
        non_ground_mask = xyz[:, 2] > z_thresh
        return xyz[non_ground_mask], xyz[~non_ground_mask]

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def remove_ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Rimuove i punti del suolo.

        Replica esattamente la sequenza di RaDelft::prepare_lidar_pointcloud:
            1. remove_ground_points_patchwork (Patchwork++)
            2. Tieni solo xyz (colonne 0:3)
            3. Filtra punti con z > -2  (cleaning_ego_car è separato)

        Args:
            points: (N, 3+) — [x, y, z, ...] in coordinate cartesiane.

        Returns:
            non_ground: (M, K) — punti non-suolo (K = numero colonne originali).
            ground:     (G, K) — punti suolo.
        """
        if len(points) == 0:
            return points.copy(), points[:0].copy()

        if PATCHWORKPP_AVAILABLE:
            non_ground_xyz, ground_xyz = self._patchwork_remove(points)
        else:
            non_ground_xyz, ground_xyz = self._height_fallback(points)

        # Applica filtro z > -2 come in RaDelft (rimuove punti sotto strada)
        # REF: data_preparation.py line: lidar_point_cloud = lidar_point_cloud[lidar_point_cloud[:, 2] > -2]
        non_ground_xyz = non_ground_xyz[non_ground_xyz[:, 2] > -2.0]

        return non_ground_xyz, ground_xyz

    def remove_ground_with_ego(
        self, points: np.ndarray, ego_x_max: float = 1.0, ego_y_max: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rimozione suolo + rimozione punti carrozzeria del veicolo.

        REF: RaDelft::cleaning_ego_car
            Remove points with x<1 AND |y|<1 (punti del tetto del veicolo).

        Args:
            points: (N, 3+).
            ego_x_max: soglia x per l'ego car (default 1.0 m).
            ego_y_max: soglia |y| per l'ego car (default 1.0 m).

        Returns:
            clean: (M, K) — punti senza suolo e senza ego car.
            ground: (G, K).
        """
        non_ground, ground = self.remove_ground(points)

        if len(non_ground) == 0:
            return non_ground, ground

        # Rimuovi punti ego car: x < ego_x_max AND |y| < ego_y_max
        # REF: cleaning_ego_car in data_preparation.py
        ego_mask = (non_ground[:, 0] < ego_x_max) & (np.abs(non_ground[:, 1]) < ego_y_max)
        clean = non_ground[~ego_mask]
        return clean, ground
