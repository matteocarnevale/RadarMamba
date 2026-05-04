"""
CFAR-Guided Spatial Attention (CGSA) Module
=============================================
REF: Paper Section 3.4.1 + Fig. 1(e)

Flusso:
  1. RAD map [B, D, R, A] → max su D → power map [B, 1, R, A]
  2. 2D CA-CFAR → maschera di attenzione ∈ [0, 1] su piano RA
  3. Attenzione moltiplicata element-wise con feature RM Block
     out = rm_features * (1 + γ * attention_map)

Il CFAR implementato con avg_pool2d differenziabile (come in cfar.py)
permette al modello di back-propagare attraverso la maschera.

Un parametro learnable γ (inizializzato a 0.5) bilancia il contributo
dell'attenzione CFAR — il modello impara a pesarla.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGSAModule(nn.Module):
    """
    CFAR-Guided Spatial Attention.

    Input:
        rm_features: (B, C, R, A)
        rad_map:     (B, D, R, A) — RAD map, canali = Doppler bins
    Output:
        out: (B, C, R, A)
    """

    def __init__(
        self,
        in_channels:    int,
        cfar_guard:     int   = 4,
        cfar_reference: int   = 8,
        cfar_alpha:     float = 4.0,
    ) -> None:
        super().__init__()
        self.guard     = cfar_guard
        self.reference = cfar_reference
        self.cfar_alpha = cfar_alpha

        # Learnable scale γ — bilancia il peso della CFAR attention
        # Inizializzato a piccolo valore per non distorcere le feature iniziali
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # Conv learnable per raffinare l'attention map CFAR (soft version)
        # Permette al modello di imparare un'attenzione più fine
        self.refine = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------

    def _cfar_attention(self, power_map: torch.Tensor) -> torch.Tensor:
        """
        Calcola la mappa di attenzione CFAR differenziabile su (B, 1, R, A).

        Usa avg_pool2d per stimare la media di riferimento.
        REF: cfar.py::ca_cfar_2d_torch

        Returns:
            attention: (B, 1, R, A) float in (0, ∞), poi clampato via refine.
        """
        g, r = self.guard, self.reference
        outer_size = 2 * (g + r) + 1
        inner_size = 2 * g + 1

        outer_avg = F.avg_pool2d(
            power_map, kernel_size=outer_size, stride=1, padding=g + r
        )
        inner_avg = F.avg_pool2d(
            power_map, kernel_size=inner_size, stride=1, padding=g
        )

        # Stima potenza media di riferimento (Eq. 9 del paper)
        N_outer = outer_size ** 2
        N_inner = inner_size ** 2
        N       = max(N_outer - N_inner, 1)
        ref_mean = (outer_avg * N_outer - inner_avg * N_inner) / N

        threshold = self.cfar_alpha * ref_mean.clamp_min(1e-8)
        # Soft ratio: valori > 1 dove c'è segnale forte
        attention = (power_map / threshold).clamp(0.0, 10.0)
        return attention   # (B, 1, R, A)

    # ------------------------------------------------------------------

    def forward(
        self,
        rm_features: torch.Tensor,   # (B, C, R, A)
        rad_map:     torch.Tensor,   # (B, D, R, A)
    ) -> torch.Tensor:
        """
        Args:
            rm_features: (B, C, R, A) — feature dal RM Block sul piano RA.
            rad_map:     (B, D, R, A) — RAD map (D = Doppler bins).

        Returns:
            out: (B, C, R, A) — feature raffinate dall'attenzione CFAR.
        """
        # 1. Proietta RAD su piano RA: max su D (target più energetico)
        power_map = rad_map.max(dim=1, keepdim=True).values   # (B, 1, R, A)

        # 2. CFAR differenziabile → soft attention map
        cfar_raw = self._cfar_attention(power_map)             # (B, 1, R, A)

        # 3. Raffina con conv learnable + sigmoid → ∈ (0, 1)
        attention = self.refine(cfar_raw)                       # (B, 1, R, A)

        # 4. Modula le feature: residual-scaling (preserva info originale)
        # out = rm_features * (1 + γ * attention)
        out = rm_features * (1.0 + self.gamma * attention)
        return out
