"""
Focal Loss per classificazione di occupancy binaria
=====================================================
REF: Paper Section 4.2:
     "We use focal loss (α=0.995, γ=2) for optimization."

     FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
     dove:
         p_t   = p   se y=1, altrimenti 1-p
         α_t   = α   se y=1, altrimenti 1-α

Il valore α=0.995 è molto alto perché la griglia di occupancy è
molto sparsa (pochi voxel occupati = classe positiva rara).
α alto bilancia il forte sbilanciamento tra classi.

NOTA STORICA (RaDelft GitHub changelog, 30-04-2025):
    "Corrected error in loss hyperparameter alpha=0.99 in the function
     radarcube_lidarcube_loss_time in data_preparation/data_preparation.py."
    → Il repo RaDelft originale usava 0.99 per errore; il paper Radar-Mamba
      usa 0.995. Noi usiamo 0.995.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss per occupancy prediction.

    Input:  logits (B, R, A, E) — output del modello PRIMA del sigmoid
    Target: binary (B, R, A, E) float32 con valori {0.0, 1.0}
    """

    def __init__(
        self,
        alpha: float = 0.995,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha:     peso per la classe positiva (paper: 0.995).
            gamma:     focusing parameter (paper: 2).
            reduction: "mean" | "sum" | "none".
        """
        super().__init__()
        assert 0.0 < alpha < 1.0, "alpha deve essere in (0, 1)"
        assert gamma >= 0.0
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcola la Focal Loss.

        Args:
            logits:  (B, R, A, E) — logits non-sigmoid del modello.
            targets: (B, R, A, E) float32 — ground truth {0.0, 1.0}.

        Returns:
            loss: scalar (se reduction="mean" o "sum") o stesso shape (se "none").
        """
        assert logits.shape == targets.shape, \
            f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"

        # Probabilità predetta: p = sigmoid(logits)
        # Usiamo binary_cross_entropy_with_logits per stabilità numerica
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # bce = -log(p_t) → (B, R, A, E)

        # p_t: probabilità della classe corretta
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # α_t: peso di classe
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focusing factor: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Focal Loss = -α_t * (1 - p_t)^γ * log(p_t) = α_t * focal_weight * bce
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction}"
