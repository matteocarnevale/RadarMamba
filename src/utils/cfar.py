"""
2D CA-CFAR (Cell-Averaging CFAR)
==================================
Implementazione numpy e torch del 2D CA-CFAR usato in CGSA.

REF: Paper Section 3.4.1, Eq. (9):
     T(i,j) = α * (1/N) * Σ_{(m,n)∈Ω} P(m,n)
     dove:
       α    = scaling factor (determinato dal false alarm rate)
       P(m,n) = potenza della reference cell (m,n)
       Ω    = reference cells (finestra meno guard cells e cell under test)
       N    = numero di reference cells

In CGSA, CFAR non serve per detection rigida ma come soft attention:
la mappa T(i,j) normalizzata viene usata come prior di attenzione.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Implementazione NumPy (usata nel preprocessing offline)
# ------------------------------------------------------------------

def ca_cfar_2d_numpy(
    power_map: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    threshold_factor: float,
) -> np.ndarray:
    """
    2D CA-CFAR su power map 2D (implementazione NumPy, per preprocessing).

    Args:
        power_map:        np.ndarray, shape (H, W) — mappa di potenza (es. RA plane).
        guard_cells:      numero di celle di guardia attorno alla CUT.
        reference_cells:  numero di celle di riferimento su ogni lato.
        threshold_factor: α (scaling factor, Eq. 9).

    Returns:
        detection_map: np.ndarray, shape (H, W) bool — True dove segnale > soglia.

    Note:
        - La finestra totale ha lato: 2*(guard_cells + reference_cells) + 1
        - Le celle di guardia vengono escluse dal calcolo della media
        - La CUT (cell under test) è al centro
    """
    H, W = power_map.shape
    detection_map = np.zeros((H, W), dtype=bool)

    g = guard_cells
    r = reference_cells
    total = g + r   # raggio della finestra esterna

    for i in range(total, H - total):
        for j in range(total, W - total):
            # Finestra esterna (reference + guard)
            outer = power_map[i - total:i + total + 1, j - total:j + total + 1]
            # Finestra interna (solo guard + CUT)
            inner = power_map[i - g:i + g + 1, j - g:j + g + 1]

            N_outer = (2 * total + 1) ** 2
            N_inner = (2 * g + 1) ** 2
            N = N_outer - N_inner

            if N == 0:
                continue

            # Somma reference (escludi guard e CUT) — Eq. (9)
            ref_sum = outer.sum() - inner.sum()
            threshold = threshold_factor * ref_sum / N

            detection_map[i, j] = power_map[i, j] > threshold

    return detection_map


# ------------------------------------------------------------------
# Implementazione PyTorch (usata in CGSA durante il forward pass)
# ------------------------------------------------------------------

def ca_cfar_2d_torch(
    power_map: torch.Tensor,
    guard_cells: int,
    reference_cells: int,
    threshold_factor: float,
    soft: bool = True,
) -> torch.Tensor:
    """
    2D CA-CFAR differenziabile su tensore 2D (H, W) — usata in CGSA.

    Implementazione tramite avg_pool2d: stima la media di riferimento
    come differenza tra pool esterno e pool interno.

    Args:
        power_map:        (H, W) — mappa di potenza.
        guard_cells:      celle di guardia.
        reference_cells:  celle di riferimento.
        threshold_factor: α.
        soft: se True, ritorna una mappa soft in [0,1] (rapporto segnale/soglia).
              se False, ritorna mappa binaria {0., 1.}.

    Returns:
        attention: (H, W) float32.

    Note:
        Questa implementazione usa avg_pool2d per calcolare la media
        di riferimento in modo vettorizzato (senza loop espliciti).
        È differenziabile ma non esattamente equivalente al CFAR esatto
        perché i bordi dell'immagine sono gestiti diversamente.
    """
    H, W = power_map.shape
    g = guard_cells
    r = reference_cells

    # Aggiungi dimensioni batch e canale per avg_pool2d
    x = power_map.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    # Media nella finestra esterna (reference + guard + CUT)
    outer_size = 2 * (g + r) + 1
    outer_avg  = F.avg_pool2d(
        x,
        kernel_size=outer_size,
        stride=1,
        padding=g + r,
    ).squeeze(0).squeeze(0)   # (H, W)

    # Media nella finestra interna (guard + CUT)
    inner_size = 2 * g + 1
    inner_avg  = F.avg_pool2d(
        x,
        kernel_size=inner_size,
        stride=1,
        padding=g,
    ).squeeze(0).squeeze(0)   # (H, W)

    # Numero di celle (approssimato per sempliciità — gestione bordi non esatta)
    N_outer = outer_size ** 2
    N_inner = inner_size ** 2
    N       = N_outer - N_inner

    # Stima della media di riferimento (Eq. 9)
    ref_mean = (outer_avg * N_outer - inner_avg * N_inner) / max(N, 1)

    # Soglia
    threshold = threshold_factor * ref_mean   # (H, W)

    if soft:
        # Attenzione soft: rapporto P / T (clip in [0, ∞])
        eps = 1e-8
        attention = (power_map / (threshold + eps)).clamp(0.0, 1.0)
    else:
        attention = (power_map > threshold).float()

    return attention.to(power_map.dtype)
