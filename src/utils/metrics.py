"""
Metriche di valutazione per point cloud enhancement
=====================================================
REF: Paper Section 4.2:
     "We use Chamfer Distance (CD) and Hausdorff Distance (HD).
      These measure the bidirectional average and maximum distances
      between corresponding nearest points in the two point clouds."

     "We also evaluate Unidirectional Chamfer Distance (UCD) and
      Unidirectional Hausdorff Distance (UHD), focusing on distances
      from LiDAR to enhanced radar point clouds to assess the model's
      ability to model occluded objects."

Nota sulla direzione UCD/UHD:
    UCD = distanza da LiDAR → radar (enhanced)
    → misura quanto il radar copre il LiDAR
    Non l'inverso (radar→LiDAR sarebbe trivialmente bassa se il radar
    genera punti sovrabbondanti vicini al LiDAR).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import KDTree


# ------------------------------------------------------------------
# Implementazioni NumPy (per valutazione offline)
# ------------------------------------------------------------------

def chamfer_distance_numpy(
    pred_pts: np.ndarray,
    gt_pts:   np.ndarray,
) -> float:
    """
    Chamfer Distance (CD) bidirezionale.

    CD = mean_{p ∈ pred} min_{g ∈ gt}   ||p-g||² +
         mean_{g ∈ gt}   min_{p ∈ pred} ||g-p||²

    Args:
        pred_pts: (N, 3) — punti predetti (radar enhanced).
        gt_pts:   (M, 3) — punti ground truth (LiDAR).

    Returns:
        cd: float — Chamfer Distance media.
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_gt   = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    # pred → gt
    dist_p2g, _ = tree_gt.query(pred_pts,   k=1)
    # gt → pred
    dist_g2p, _ = tree_pred.query(gt_pts, k=1)

    cd = dist_p2g.mean() + dist_g2p.mean()
    return float(cd)


def hausdorff_distance_numpy(
    pred_pts: np.ndarray,
    gt_pts:   np.ndarray,
) -> float:
    """
    Hausdorff Distance (HD) bidirezionale.

    HD = max(max_{p ∈ pred} min_{g ∈ gt} ||p-g||,
             max_{g ∈ gt}   min_{p ∈ pred} ||g-p||)

    Args:
        pred_pts: (N, 3)
        gt_pts:   (M, 3)

    Returns:
        hd: float.
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_gt   = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    dist_p2g, _ = tree_gt.query(pred_pts, k=1)
    dist_g2p, _ = tree_pred.query(gt_pts, k=1)

    hd = max(dist_p2g.max(), dist_g2p.max())
    return float(hd)


def unidirectional_chamfer_numpy(
    pred_pts: np.ndarray,
    gt_pts:   np.ndarray,
) -> float:
    """
    Unidirectional Chamfer Distance (UCD): solo LiDAR → radar.

    UCD = mean_{g ∈ gt} min_{p ∈ pred} ||g-p||

    Misura quanto il radar copre il LiDAR (obiettivo del paper).

    Args:
        pred_pts: (N, 3) — radar enhanced (predetto).
        gt_pts:   (M, 3) — LiDAR (ground truth).

    Returns:
        ucd: float.
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_pred = KDTree(pred_pts)
    dist_g2p, _ = tree_pred.query(gt_pts, k=1)
    return float(dist_g2p.mean())


def unidirectional_hausdorff_numpy(
    pred_pts: np.ndarray,
    gt_pts:   np.ndarray,
) -> float:
    """
    Unidirectional Hausdorff Distance (UHD): solo LiDAR → radar.

    UHD = max_{g ∈ gt} min_{p ∈ pred} ||g-p||

    Args:
        pred_pts: (N, 3)
        gt_pts:   (M, 3)

    Returns:
        uhd: float.
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_pred = KDTree(pred_pts)
    dist_g2p, _ = tree_pred.query(gt_pts, k=1)
    return float(dist_g2p.max())


# ------------------------------------------------------------------
# Wrapper per valutazione su un batch di campioni
# ------------------------------------------------------------------

class PointCloudMetrics:
    """
    Calcola CD, HD, UCD, UHD su batch di point cloud e fa la media.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._scores: dict[str, list[float]] = {
            "CD": [], "HD": [], "UCD": [], "UHD": [],
            "n_pred_points": [], "n_gt_points": [],
        }

    def update(
        self,
        pred_pts: np.ndarray,
        gt_pts:   np.ndarray,
    ) -> dict[str, float]:
        """
        Aggiunge un campione alle metriche.

        Args:
            pred_pts: (N, 3) — punti predetti.
            gt_pts:   (M, 3) — punti ground truth LiDAR.

        Returns:
            dict con le metriche per questo campione.
        """
        cd  = chamfer_distance_numpy(pred_pts, gt_pts)
        hd  = hausdorff_distance_numpy(pred_pts, gt_pts)
        ucd = unidirectional_chamfer_numpy(pred_pts, gt_pts)
        uhd = unidirectional_hausdorff_numpy(pred_pts, gt_pts)

        self._scores["CD"].append(cd)
        self._scores["HD"].append(hd)
        self._scores["UCD"].append(ucd)
        self._scores["UHD"].append(uhd)
        self._scores["n_pred_points"].append(len(pred_pts))
        self._scores["n_gt_points"].append(len(gt_pts))

        return {"CD": cd, "HD": hd, "UCD": ucd, "UHD": uhd}

    def compute(self) -> dict[str, float]:
        """
        Restituisce le metriche mediate su tutti i campioni aggiornati.
        """
        return {
            k: float(np.mean(v)) if v else float("nan")
            for k, v in self._scores.items()
        }

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"CD={m['CD']:.4f}  HD={m['HD']:.4f}  "
            f"UCD={m['UCD']:.4f}  UHD={m['UHD']:.4f}  "
            f"N_pts={m['n_pred_points']:.0f}"
        )
