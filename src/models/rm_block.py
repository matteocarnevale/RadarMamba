"""
Radar-Mamba (RM) Block
========================
REF: Paper Section 3.3.3 + Fig. 1(b) — Equations (5)–(8)

  X_conv  = Linear( Σ_{k∈{3,5,7}} DWConv_{k×k}(X_in) )          (Eq. 5)
  X1,X2,X3,X4 = Split(X_conv)                                     (Eq. 6)
  X_Ri = RHSS_i(X_i) * W(X_i),   i = 1,2,3,4                     (Eq. 7)
  X_out = Linear[LN[Concat(X_R1,...,X_R4) + X_conv]]              (Eq. 8)

Note sul gating W(X_i):
    Il gate W(X_i) è una proiezione lineare + SiLU dello stesso gruppo X_i.
    Produce pesi scalari per ogni feature — meccanismo GLU (Gated Linear Unit).
    SiLU (σ(x)·x) è usata da Mamba internamente, quindi consistente.

I/O: [B, C, H, W] → [B, C, H, W]   (H=Range, W=Azimuth nel BEV)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rhss import RHSSModule


class RMBlock(nn.Module):
    """Radar-Mamba Block — implementazione completa delle Eq. (5)–(8)."""

    def __init__(
        self,
        in_channels:  int,
        expand_ratio: float = 2.0,
        n_groups:     int   = 4,
        conv_kernels: list[int] | None = None,
        ssm_d_state:  int = 16,
        ssm_d_conv:   int = 4,
        ssm_expand:   int = 2,
    ) -> None:
        """
        Args:
            in_channels:  C — canali input.
            expand_ratio: fattore di espansione per lo spazio interno.
            n_groups:     numero di gruppi per lo split (paper: 4).
            conv_kernels: kernel multi-scala (paper: [3, 5, 7]).
            ssm_*:        parametri Mamba per ogni RHSS.
        """
        super().__init__()

        if conv_kernels is None:
            conv_kernels = [3, 5, 7]

        self.in_channels = in_channels
        self.n_groups    = n_groups

        # inner_channels deve essere divisibile per n_groups
        inner_raw    = int(in_channels * expand_ratio)
        inner        = (inner_raw // n_groups) * n_groups
        self.inner   = inner
        self.g_ch    = inner // n_groups   # canali per gruppo

        # ── Linear di proiezione input → inner_channels (Eq. 5, prima parte) ──
        self.input_proj = nn.Linear(in_channels, inner, bias=False)

        # ── Multi-scale depthwise conv (Eq. 5) ─────────────────────────────────
        # DWConv_{k×k} : conv con groups=inner (depthwise)
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner, inner, kernel_size=k, padding=k // 2, groups=inner, bias=False),
                nn.BatchNorm2d(inner),
            )
            for k in conv_kernels
        ])

        # Linear per proiettare dopo la somma delle DWConv
        self.conv_proj = nn.Linear(inner, inner, bias=False)

        # ── 4 moduli RHSS (uno per gruppo, Eq. 7) ────────────────────────────
        self.rhss = nn.ModuleList([
            RHSSModule(self.g_ch, ssm_d_state, ssm_d_conv, ssm_expand)
            for _ in range(n_groups)
        ])

        # ── Gate W(X_i) per ogni gruppo — SiLU gate (Eq. 7) ─────────────────
        self.gates = nn.ModuleList([
            nn.Linear(self.g_ch, self.g_ch, bias=False)
            for _ in range(n_groups)
        ])

        # ── Output: LN + Linear (Eq. 8) ──────────────────────────────────────
        self.layer_norm  = nn.LayerNorm(inner)
        self.output_proj = nn.Linear(inner, in_channels, bias=False)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ── Proiezione input (parte di Eq. 5) ─────────────────────────────────
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)     # (B*H*W, C)
        x_proj = self.input_proj(x_flat)                           # (B*H*W, inner)
        x_proj = x_proj.reshape(B, H, W, self.inner).permute(0, 3, 1, 2)  # (B, inner, H, W)

        # ── Multi-scale DWConv + somma + proiezione (Eq. 5) ─────────────────
        conv_sum = sum(dw(x_proj) for dw in self.dw_convs)        # (B, inner, H, W)

        c_flat  = conv_sum.permute(0, 2, 3, 1).reshape(B * H * W, self.inner)
        X_conv  = self.conv_proj(c_flat).reshape(B, H, W, self.inner).permute(0, 3, 1, 2)
        # X_conv: (B, inner, H, W)

        # ── Split in n_groups gruppi (Eq. 6) ─────────────────────────────────
        # split su dim=1 (canali)
        groups = X_conv.split(self.g_ch, dim=1)   # n_groups × (B, g_ch, H, W)

        # ── RHSS_i(X_i) * W(X_i) per ogni gruppo (Eq. 7) ───────────────────
        X_R_list = []
        for i, (grp, rhss_mod, gate_lin) in enumerate(zip(groups, self.rhss, self.gates)):
            # RHSS: (B, g_ch, H, W) → (B, g_ch, H, W)
            rhss_out = rhss_mod(grp)

            # Gate W(X_i): linear + SiLU
            g_flat = grp.permute(0, 2, 3, 1).reshape(B * H * W, self.g_ch)
            gate   = F.silu(gate_lin(g_flat))                # (B*H*W, g_ch)
            gate   = gate.reshape(B, H, W, self.g_ch).permute(0, 3, 1, 2)   # (B, g_ch, H, W)

            X_R_list.append(rhss_out * gate)

        # ── Concat + residuo X_conv + LN + Linear (Eq. 8) ─────────────────
        X_cat = torch.cat(X_R_list, dim=1)    # (B, inner, H, W)
        X_res = X_cat + X_conv                # residuo dalla somma DWConv

        # LN in channel-last
        res_flat  = X_res.permute(0, 2, 3, 1).reshape(B * H * W, self.inner)
        ln_flat   = self.layer_norm(res_flat)                   # (B*H*W, inner)
        out_flat  = self.output_proj(ln_flat)                   # (B*H*W, C)
        out_2d    = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Skip connection con l'input originale
        return out_2d + x
