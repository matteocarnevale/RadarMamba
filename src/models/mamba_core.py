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
     Installazione: pip install mamba-ssm causal-conv1d

Il modulo Mamba prende sequenze [B, L, D] e restituisce [B, L, D].
In RHSS lo usiamo per processare ogni sequenza ottenuta dai diversi
pattern di scansione (VMamba, zigzag, inside-out).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Prova a importare mamba-ssm (richiede CUDA)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaSSM(nn.Module):
    """
    Wrapper attorno al modulo Mamba di mamba-ssm.

    Se mamba-ssm non è disponibile (CPU o CUDA non compatibile),
    usa un MambaFallback basato su LSTM — più lento ma funzionale
    per sviluppo e debugging.

    Args:
        d_model: dimensione delle feature (D nel paper).
        d_state: dimensione dello stato SSM (N nel paper, default 16).
        d_conv:  larghezza della convoluzione locale (default 4).
        expand:  fattore di espansione del blocco (default 2).
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

        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._backend = "mamba_ssm"
        else:
            import warnings
            warnings.warn(
                "mamba-ssm non disponibile. Uso LSTM come fallback (più lento).\n"
                "Installa mamba-ssm con: pip install mamba-ssm causal-conv1d",
                RuntimeWarning,
                stacklevel=2,
            )
            self.ssm = MambaFallback(d_model, d_state, expand)
            self._backend = "lstm_fallback"

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
