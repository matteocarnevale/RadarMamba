"""
Decoder — BEV → Occupancy [B, R, A, E]
========================================
REF: Paper Section 3.3.2 + Section 4.2:

"The decoder progressively upsamples features to generate a 3D occupancy
 grid [R, A, E], where each voxel indicates object presence."

"Our regression head, which consists of two standard CNN layers with
 sigmoid activation, outputs height classification for each BEV grid point
 as a 3D occupancy grid matching the input resolution."

Strategia "height classification":
    Per ogni cella BEV (r, a), predici la probabilità di occupazione
    per ognuno degli E bin di elevation — è una regressione multi-etichetta
    (non softmax: ogni bin è indipendente, uso BCE/FocalLoss binaria).

Il decoder è intenzionalmente semplice (2 CNN head) per rispettare
il budget di 2.4M parametri totali.
Con skip connections dall'encoder il decoder può essere più profondo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyDecoder(nn.Module):
    """
    Decoder BEV → occupancy logits (B, R, A, E).
    Input: feature BEV (B, C, R, A) dal bottleneck.
    """

    def __init__(
        self,
        in_channels:      int = 64,
        n_elevation_bins: int = 11,
        mid_channels:     list[int] | None = None,
        use_skip:         bool = True,
        skip_channels:    list[int] | None = None,
    ) -> None:
        """
        Args:
            in_channels:      C — canali input dal bottleneck (64).
            n_elevation_bins: E — bin di elevation da predire.
            mid_channels:     canali intermedi del decoder body.
            use_skip:         se True, incorpora skip connections dall'encoder.
            skip_channels:    canali delle skip connections (se use_skip).
        """
        super().__init__()

        if mid_channels is None:
            mid_channels = [64, 32]

        self.use_skip = use_skip
        self.n_elevation_bins = n_elevation_bins

        # ── Corpo del decoder (conv 2D progressivo) ───────────────────
        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(mid_channels):
            # Se use_skip, il primo livello aggiunge skip_channels[i]
            in_ch = prev_ch + (skip_channels[i] if use_skip and skip_channels and i < len(skip_channels) else 0)
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            ]
            prev_ch = ch
        self.body = nn.Sequential(*layers)

        # ── Head: 2 × Conv + niente sigmoid (sigmoide applicato nella loss) ──
        # Paper: "two standard CNN layers with sigmoid activation"
        # Usiamo la sigmoid nella loss (BCE with logits) per stabilità numerica.
        self.head = nn.Sequential(
            nn.Conv2d(prev_ch, prev_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prev_ch),
            nn.GELU(),
            nn.Conv2d(prev_ch, n_elevation_bins, kernel_size=1),   # logits
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        bev_feat:     torch.Tensor,
        skip_feats:   list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            bev_feat:   (B, C, R, A)
            skip_feats: lista di tensori dall'encoder per skip connections
                        (opzionale, stessa scala spaziale di bev_feat).

        Returns:
            logits: (B, R, A, E) — logits non-sigmoid.
                    Per ottenere occupancy: torch.sigmoid(logits).
        """
        B, C, R, A = bev_feat.shape
        x = bev_feat

        # Incorpora skip connections se disponibili
        if self.use_skip and skip_feats:
            for skip in skip_feats:
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)

        x = self.body(x)         # (B, mid[-1], R, A)
        logits = self.head(x)    # (B, E, R, A)

        # Trasponi → (B, R, A, E) per matchare il GT shape
        return logits.permute(0, 2, 3, 1).contiguous()
