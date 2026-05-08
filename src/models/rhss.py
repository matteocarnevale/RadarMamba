"""
Radar Hybrid Selective Scan (RHSS) Module
==========================================
REF: Paper Section 3.3.4 + Fig. 1(d)

8 scanning patterns su una feature map 2D [B, C, H, W] (H=Range, W=Azimuth):

  Pattern 0–3: Global scans (VMamba-style)
      0: row-major raster              (top-left → bottom-right)
      1: reverse row-major             (bottom-right → top-left)
      2: column-major                  (top-left → bottom-right, column first)
      3: reverse column-major          (bottom-right → top-left, column first)

  Pattern 4–5: Zigzag local scans (LocalMamba-style)
      4: horizontal snake              (row 0 L→R, row 1 R→L, ...)
      5: vertical snake                (col 0 T→B, col 1 B→T, ...)

  Pattern 6–7: Inside-out local scans (custom, radar-specific)
      6: inside-out by azimuth center  (sort by |a - A//2|, then r)
         → "targets in center FoV detected first"
      7: inside-out by 2D center dist  (sort by sqrt(r² + (a-A//2)²))
         → "near + center first"

Paper: "This diverse scanning strategy greatly improves the model's ability to
extract global and local features from radar point clouds."
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.mamba_core import MambaSSM


# ------------------------------------------------------------------
# Pattern generators — produzione degli indici flat (H*W,) su CPU
# ------------------------------------------------------------------

def _make_scan_patterns(H: int, W: int) -> List[torch.Tensor]:
    """
    Genera gli 8 ordini di scansione come tensori di indici flat (H*W,) su CPU.
    I tensori vengono successivamente spostati sul device del tensore di input.
    """
    L = H * W

    # ── Pattern 0: row-major ──────────────────────────────────────────
    p0 = torch.arange(L)

    # ── Pattern 1: reverse row-major ─────────────────────────────────
    p1 = torch.arange(L - 1, -1, -1)

    # ── Pattern 2: column-major ───────────────────────────────────────
    # Ordine: (0,0),(1,0),(2,0),...,(H-1,0),(0,1),(1,1),...
    i = torch.arange(H).unsqueeze(1).expand(H, W)   # [H, W]
    j = torch.arange(W).unsqueeze(0).expand(H, W)   # [H, W]
    p2 = (j * H + i).reshape(-1)                     # [L] — column-major flat

    # ── Pattern 3: reverse column-major ──────────────────────────────
    p3 = torch.flip(p2, dims=[0])

    # ── Pattern 4: horizontal snake ───────────────────────────────────
    rows = []
    for row in range(H):
        cols = torch.arange(W)
        if row % 2 == 1:
            cols = torch.flip(cols, dims=[0])
        rows.append(row * W + cols)
    p4 = torch.cat(rows)

    # ── Pattern 5: vertical snake ─────────────────────────────────────
    cols = []
    for col in range(W):
        idxs = torch.arange(H)
        if col % 2 == 1:
            idxs = torch.flip(idxs, dims=[0])
        cols.append(idxs * W + col)
    p5 = torch.cat(cols)

    # ── Pattern 6: inside-out by azimuth center ──────────────────────
    # Paper: "center region corresponding to open roads" → scan center FoV first
    # Sort primary: |a - W//2|, secondary: i (ascending range = near first)
    az_center = W // 2
    i_g = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    j_g = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    az_dist = (j_g - az_center).abs()
    # Lexicographic key: scale az_dist so it dominates over range
    sort_key6 = az_dist * H + i_g
    p6 = torch.argsort(sort_key6, stable=True)

    # ── Pattern 7: inside-out by 2D Euclidean center distance ────────
    # Sort by distance from (0, W//2) — near range + front azimuth first
    sort_key7 = torch.sqrt(i_g ** 2 + (j_g - az_center) ** 2)
    p7 = torch.argsort(sort_key7, stable=True)

    return [p0, p1, p2, p3, p4, p5, p6, p7]


@lru_cache(maxsize=16)
def _cached_patterns(H: int, W: int) -> List[torch.Tensor]:
    """Cache dei pattern di scansione per (H, W) fissa (evita ricalcolo ad ogni forward)."""
    return _make_scan_patterns(H, W)


# ------------------------------------------------------------------
# RHSS Module
# ------------------------------------------------------------------

class RHSSModule(nn.Module):
    """
    Radar Hybrid Selective Scan Module.

    Input:  [B, C, H, W]   (H = Range, W = Azimuth nel frame BEV)
    Output: [B, C, H, W]

    Applica 8 SSM indipendenti su 8 diversi ordini di scansione,
    somma i risultati e applica proiezione di output + residuo.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        env_n_patterns = int(os.getenv("RHSS_N_PATTERNS", "8"))
        self.n_patterns = max(1, min(8, env_n_patterns))

        # SSM indipendenti (uno per pattern). Default: 8 (paper).
        self.ssm_list = nn.ModuleList([
            MambaSSM(d_model, d_state, d_conv, expand)
            for _ in range(self.n_patterns)
        ])

        # Proiezione output: torna a d_model dopo aggregazione
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    def _scan_and_process(
        self,
        x:       torch.Tensor,   # (B, C, H, W)
        indices: torch.Tensor,   # (L,) indici flat sull'order corretto
        ssm:     MambaSSM,
    ) -> torch.Tensor:
        """
        Appiattisce x seguendo l'ordine 'indices', processa con SSM,
        poi re-ordina nell'ordine originale.

        Returns:
            y: (B, C, H, W) — feature processate in quest'ordine di scansione.
        """
        B, C, H, W = x.shape
        L = H * W

        # Riordina lungo la scansione: [B, C, L] → reorder → [B, L, C]
        x_flat = x.reshape(B, C, L)            # (B, C, L)
        x_scan = x_flat[:, :, indices]          # (B, C, L) — riordinato
        x_seq  = x_scan.permute(0, 2, 1)       # (B, L, C)

        # SSM: [B, L, C] → [B, L, C]
        y_seq = ssm(x_seq)

        # Ripristina l'ordine originale
        inv_idx = torch.argsort(indices)
        y_back  = y_seq.permute(0, 2, 1)       # (B, C, L)
        y_orig  = y_back[:, :, inv_idx]         # (B, C, L)

        return y_orig.reshape(B, C, H, W)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        dev = x.device

        # Ottieni pattern di scansione (cached, su CPU poi sposta)
        patterns_cpu = _cached_patterns(H, W)
        patterns = [p.to(dev) for p in patterns_cpu[: self.n_patterns]]

        # Applica ogni SSM sul proprio pattern e accumula
        agg = torch.zeros_like(x)
        for pattern, ssm in zip(patterns, self.ssm_list):
            if self.training and self.use_checkpoint and x.requires_grad:
                # Recompute each scan in backward to reduce activation memory.
                scan_out = checkpoint(
                    lambda inp, _pattern=pattern, _ssm=ssm: self._scan_and_process(inp, _pattern, _ssm),
                    x,
                    use_reentrant=False,
                )
            else:
                scan_out = self._scan_and_process(x, pattern, ssm)
            agg = agg + scan_out

        # Media delle uscite dei pattern usati (paper: 8).
        agg = agg / float(self.n_patterns)

        # Proiezione output: in channel-last per Linear
        agg_flat = agg.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, L, C)
        out_flat = self.out_proj(agg_flat)                          # (B, L, C)

        # Residuo + LayerNorm (su channel-last)
        out_ln = self.norm(out_flat + x.permute(0, 2, 3, 1).reshape(B, H * W, C))

        # Ritorna a (B, C, H, W)
        return out_ln.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
