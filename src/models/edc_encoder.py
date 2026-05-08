"""
Elevation-Doppler Compression (EDC) Encoder
=============================================
REF: Paper Section 3.3.2

Doppio ramo ispirato a SlowFast Network:
  - Ramo Elevation: comprime la dimensione E (struttura 3D spaziale)
  - Ramo Doppler:   cattura variazioni temporali Doppler tra i 3 frame

Input:  (B, 6, R, A, E)   — radar cube fuso temporalmente
Output: (B, out_channels=64, R, A)   — feature map BEV

Canali input:
    [0]=I_t, [1]=D_t, [2]=I_{t-1}, [3]=D_{t-1}, [4]=I_{t-2}, [5]=D_{t-2}
    (I=intensità, D=Doppler, t=frame corrente, t-1/t-2=frame precedenti)

Il ramo Doppler usa solo i canali D: indici [1, 3, 5].
Il ramo Elevation usa tutti i 6 canali (include la struttura completa).

AdaptiveAvgPool3D((1, R, A)) comprime E a 1 indipendentemente da E_in,
quindi funziona sia per RADIal (E=11) che per RaDelft (E=34).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDCEncoder(nn.Module):
    """
    Elevation-Doppler Compression Encoder.
    Input:  (B, 6, R, A, E) — radar cube
    Output: (B, 64, R, A)   — feature BEV
    """

    def __init__(
        self,
        in_channels:  int = 6,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self._warned_cudnn_fallback = False
        mid = out_channels // 2    # 32 per ramo

        # ── Ramo Elevation ─────────────────────────────────────────────
        # Tratta input come (B, C=6, E, R, A) → Conv3D lungo (E, R, A)
        # → AdaptiveAvgPool3D((1, R, A)) → squeeze E → Conv2D
        self.elevation_3d = nn.Sequential(
            # Kernel (E_k, 3, 3): cattura struttura elevazione + locale RA
            nn.Conv3d(in_channels, mid, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            nn.Conv3d(mid, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
        )
        # Pool adattivo: comprime E a 1, mantiene (R, A) intatti
        self.elevation_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.elevation_2d = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )

        # ── Ramo Doppler ───────────────────────────────────────────────
        # Usa solo i 3 canali Doppler (indici 1, 3, 5) — uno per frame
        # Tratta come (B, 3, E, R, A) → Conv3D → pool → Conv2D
        n_dop = 3   # D_t, D_{t-1}, D_{t-2}
        self.doppler_3d = nn.Sequential(
            # Kernel (3, 3, 3): cattura dipendenza temporale (3 frame) + spaziale
            nn.Conv3d(n_dop, mid, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            nn.Conv3d(mid, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
        )
        self.doppler_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.doppler_2d = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )

        # ── Fusione: mid × 2 → out_channels ───────────────────────────
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    def _run_3d_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Retry Conv3D stack without cuDNN when cuDNN init fails.
        This keeps training alive on constrained GPUs where cuDNN may fail
        during the first 3D kernel initialization.
        """
        try:
            return block(x)
        except RuntimeError as exc:
            msg = str(exc)
            if x.is_cuda and "CUDNN_STATUS_NOT_INITIALIZED" in msg:
                torch.cuda.empty_cache()
                with torch.backends.cudnn.flags(enabled=False):
                    out = block(x)
                if not self._warned_cudnn_fallback:
                    warnings.warn(
                        "cuDNN initialization failed in EDC Conv3D; "
                        "falling back to non-cuDNN Conv3D kernels.",
                        RuntimeWarning,
                    )
                    self._warned_cudnn_fallback = True
                return out
            raise

    def forward(self, radar_cube: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radar_cube: (B, 6, R, A, E)
                        Nota: l'ultima dimensione è E (elevation).
                        Riorganizzo in (B, C, E, R, A) per Conv3D di PyTorch.

        Returns:
            bev: (B, 64, R, A)
        """
        B, C, R, A, E = radar_cube.shape

        # Riordina per Conv3D: (B, C, E, R, A)
        x = radar_cube.permute(0, 1, 4, 2, 3).contiguous()    # (B, 6, E, R, A)

        # ── Ramo Elevation ─────────────────────────────────────────────
        el_feat = self._run_3d_block(self.elevation_3d, x)      # (B, 32, E', R, A)
        el_feat = self.elevation_pool(el_feat).squeeze(2)       # (B, 32, R, A)
        el_feat = self.elevation_2d(el_feat)                    # (B, 32, R, A)

        # ── Ramo Doppler ───────────────────────────────────────────────
        # Estrai canali Doppler: 1, 3, 5  → (B, 3, E, R, A)
        dop = x[:, 1::2, :, :, :]                              # (B, 3, E, R, A)
        dop_feat = self._run_3d_block(self.doppler_3d, dop)     # (B, 32, E', R, A)
        dop_feat = self.doppler_pool(dop_feat).squeeze(2)       # (B, 32, R, A)
        dop_feat = self.doppler_2d(dop_feat)                    # (B, 32, R, A)

        # ── Fusione ─────────────────────────────────────────────────
        combined = torch.cat([el_feat, dop_feat], dim=1)        # (B, 64, R, A)
        bev = self.fusion(combined)                             # (B, 64, R, A)
        return bev
