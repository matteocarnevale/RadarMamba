"""
Doppler-Aware Feature Extraction Backbone
==========================================
REF: Paper Section 3.4.1 + Fig. 1(a3)

Input:  RAD map (B, D, R, A)  —  D = Doppler velocity bins
Output: Doppler features (B, out_ch, R, A)

Pipeline per ogni passo:
    RM Block → CGSA(rad_map) → repeat × n_blocks

La RAD map viene passata ad ogni CGSA per guidare l'attenzione spaziale
con informazione Doppler-noise correlation. Il RM Block estrae le feature,
CGSA le raffina sopprimendo il rumore guidato da CFAR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rm_block import RMBlock
from src.models.cgsa import CGSAModule


class DopplerBackbone(nn.Module):
    """
    Backbone Doppler-aware: proiezione D→out_ch + stack (RM Block + CGSA).
    """

    def __init__(
        self,
        in_channels:    int,          # D — Doppler bins
        out_channels:   int = 64,
        n_blocks:       int = 2,
        cfar_guard:     int = 4,
        cfar_reference: int = 8,
        cfar_alpha:     float = 4.0,
        rm_kwargs:      dict | None = None,
    ) -> None:
        """
        Args:
            in_channels:  D — Doppler bins della RAD map.
            out_channels: canali output (64 come nel paper, stesso dell'encoder).
            n_blocks:     numero di coppie (RM + CGSA).
            cfar_*:       parametri CFAR per i moduli CGSA.
            rm_kwargs:    keyword args extra per RMBlock.
        """
        super().__init__()
        if rm_kwargs is None:
            rm_kwargs = {}

        # ── Proiezione D → out_channels ─────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # ── Stack di coppie (RM Block, CGSA) ────────────────────────
        self.rm_blocks = nn.ModuleList([
            RMBlock(in_channels=out_channels, **rm_kwargs)
            for _ in range(n_blocks)
        ])
        self.cgsa_list = nn.ModuleList([
            CGSAModule(
                in_channels=out_channels,
                cfar_guard=cfar_guard,
                cfar_reference=cfar_reference,
                cfar_alpha=cfar_alpha,
            )
            for _ in range(n_blocks)
        ])

    # ------------------------------------------------------------------

    def forward(self, rad_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rad_map: (B, D, R, A) — RAD map in channel-first.
                     Note: la RAD map originale è [R, A, D]; dopo il
                     caricamento è già trasposta in (D, R, A) dal Dataset.

        Returns:
            doppler_features: (B, out_channels, R, A)
        """
        # Proietta D → out_channels
        x = self.input_proj(rad_map)   # (B, out_ch, R, A)

        # Stack (RM Block + CGSA): CGSA usa la RAD originale come guida
        for rm, cgsa in zip(self.rm_blocks, self.cgsa_list):
            x = rm(x)          # (B, out_ch, R, A)
            x = cgsa(x, rad_map)    # affina con attenzione CFAR sulla RAD originale

        return x
