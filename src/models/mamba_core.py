"""
Mamba Core — wrapper attorno a mamba-ssm
==========================================
Fornisce un'interfaccia uniforme per il modulo SSM (State Space Model)
usato in RHSS, DEF e Doppler backbone.

REF: Paper Section 3.3.1 (Preliminaries):
     h_t = A·h_{t-1} + B·x_t
     y_t = C·h_t
     con A, B, C discretizzati via ZOH e parametri selettivi (input-dependent).

Mamba (Gu & Dao, 2023):
     GitHub: https://github.com/state-spaces/mamba
     Installazione (richiede nvcc nel PATH + CUDA >= 11.6):
         export PATH=/usr/local/cuda-XX.X/bin:$PATH
         pip install mamba-ssm causal-conv1d
     Se nvcc non trovato o CUDA incompatibile (es. cu128 senza wheel):
         pip install git+https://github.com/state-spaces/mamba.git causal-conv1d

Se mamba-ssm NON è installabile, il modulo usa automaticamente un
fallback LSTM bidirezionale — stesso comportamento, performance inferiori.
Per forzare il fallback (debug senza GPU): imposta MAMBA_FORCE_FALLBACK=1.

    export MAMBA_FORCE_FALLBACK=1
    python scripts/train.py --config configs/radial.yaml
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaSSM(nn.Module):
    """
    Wrapper attorno al modulo Mamba di mamba-ssm con fallback automatico.

    Se mamba-ssm non è disponibile usa MambaFallback (LSTM bidirezionale).
    Stessa interfaccia: [B, L, D] → [B, L, D].

    Args:
        d_model: dimensione delle feature.
        d_state: dimensione dello stato SSM (default 16).
        d_conv:  larghezza convoluzione locale (default 4).
        expand:  fattore di espansione (default 2).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )            
        self._backend = "mamba_ssm"
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape (B, L, D) — sequenza di features.

        Returns:
            y: torch.Tensor, shape (B, L, D).
        """
        return self.ssm(x)


class MambaFallback(nn.Module):
    """
    Fallback basato su LSTM bidirezionale per sviluppo senza CUDA.
    Stessa interfaccia di Mamba: input [B, L, D] → output [B, L, D].

    NON usare per training finale — le performance saranno inferiori.
    Serve solo per verificare shape e logica della pipeline.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        hidden = d_model * expand
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # (B, L, hidden)
        return self.proj(out)   # (B, L, d_model)
