"""
Doppler-Elevation Fusion (DEF) Block
======================================
REF: Paper Section 3.4.2 + Fig. 1(c)

"The corresponding velocity and elevation features are first projected into
a unified feature space through linear layers and depthwise separable
convolutions. The projected features are concatenated and further processed
by the RHSS module to extract enriched hybrid features. Finally, a residual
connection with a linear layer outputs the enhanced Doppler-Elevation fused
feature tensor."

Flusso preciso:
  doppler_feat   [B, C_d, R, A] ─┐
                                  ├─ Linear+DWConv → [B, H, R, A]
  elevation_feat [B, C_e, R, A] ─┘               ─ Linear+DWConv → [B, H, R, A]
  → Concat → [B, 2H, R, A]
  → RHSS
  → Linear [2H → C_out]
  → residual(elevation_feat projected) + out
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rhss import RHSSModule


class DEFBlock(nn.Module):
    """Doppler-Elevation Fusion Block — Fig. 1(c)."""

    def __init__(
        self,
        doppler_ch:   int,
        elevation_ch: int,
        hidden_ch:    int | None = None,
        out_ch:       int | None = None,
        ssm_d_state:  int = 16,
        ssm_d_conv:   int = 4,
        ssm_expand:   int = 2,
    ) -> None:
        """
        Args:
            doppler_ch:   C_d — canali dal Doppler backbone.
            elevation_ch: C_e — canali dall'encoder principale.
            hidden_ch:    H   — spazio di proiezione condiviso (default: min(C_d,C_e)).
            out_ch:       C_out — canali output (default: elevation_ch).
            ssm_*:        parametri Mamba per l'RHSS interno.
        """
        super().__init__()

        if hidden_ch is None:
            hidden_ch = min(doppler_ch, elevation_ch)
        if out_ch is None:
            out_ch = elevation_ch

        self.hidden_ch = hidden_ch
        self.out_ch    = out_ch

        # ── Proiezione Doppler → spazio condiviso H ─────────────────────
        # Linear + DWConv (depthwise separable)
        self.dop_linear = nn.Linear(doppler_ch, hidden_ch, bias=False)
        self.dop_dw     = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1,
                      groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
        )

        # ── Proiezione Elevation → spazio condiviso H ────────────────────
        self.el_linear = nn.Linear(elevation_ch, hidden_ch, bias=False)
        self.el_dw     = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1,
                      groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
        )

        # ── RHSS su feature concatenate (2H canali) ──────────────────────
        self.rhss = RHSSModule(hidden_ch * 2, ssm_d_state, ssm_d_conv, ssm_expand)

        # ── Linear di output 2H → C_out + LN ────────────────────────────
        self.out_linear = nn.Linear(hidden_ch * 2, out_ch, bias=False)
        self.layer_norm = nn.LayerNorm(out_ch)

        # Residual: proietta elevation_feat a out_ch se C_e ≠ C_out
        self.residual_proj = (
            nn.Conv2d(elevation_ch, out_ch, kernel_size=1, bias=False)
            if elevation_ch != out_ch else nn.Identity()
        )

    # ------------------------------------------------------------------

    def _project(
        self, feat: torch.Tensor, linear: nn.Linear, dw: nn.Sequential
    ) -> torch.Tensor:
        """Linear + DWConv su [B, C, R, A]."""
        B, C, R, A = feat.shape
        # Linear su channel-last
        f_flat = feat.permute(0, 2, 3, 1).reshape(B * R * A, C)
        f_proj = linear(f_flat).reshape(B, R, A, self.hidden_ch).permute(0, 3, 1, 2)
        return dw(f_proj)   # (B, hidden_ch, R, A)

    # ------------------------------------------------------------------

    def forward(
        self,
        doppler_feat:   torch.Tensor,   # (B, C_d, R, A)
        elevation_feat: torch.Tensor,   # (B, C_e, R, A)
    ) -> torch.Tensor:
        """
        Returns:
            fused: (B, C_out, R, A)
        """
        B, _, R, A = elevation_feat.shape

        # 1. Proiezioni in spazio condiviso H
        dop_h = self._project(doppler_feat,   self.dop_linear, self.dop_dw)   # (B, H, R, A)
        el_h  = self._project(elevation_feat, self.el_linear,  self.el_dw)    # (B, H, R, A)

        # 2. Concatenazione → (B, 2H, R, A)
        combined = torch.cat([dop_h, el_h], dim=1)

        # 3. RHSS su feature ibride
        combined = self.rhss(combined)    # (B, 2H, R, A)

        # 4. Linear di output in channel-last + LN
        c_flat  = combined.permute(0, 2, 3, 1).reshape(B * R * A, self.hidden_ch * 2)
        out_flat = self.out_linear(c_flat)                                     # (B*R*A, out_ch)
        out_ln   = self.layer_norm(out_flat)
        out      = out_ln.reshape(B, R, A, self.out_ch).permute(0, 3, 1, 2)   # (B, out_ch, R, A)

        # 5. Residual connection con elevation_feat
        residual = self.residual_proj(elevation_feat)
        return out + residual
