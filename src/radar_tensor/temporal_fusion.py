"""
Temporal Fusion — 3 Frame → [R, A, E, 6]
==========================================
Aggrega i tensori [R, A, E, 2] di 3 frame consecutivi in un
unico tensore [R, A, E, 6] impilando i canali temporalmente.

REF: Paper Section 3.2:
     "A temporal fusion module aggregates elevation and velocity features
      from the previous two frames, enhancing the tensor to [R, A, E, 6]."

     Paper Figure 1(a2):
     "The model takes a processed radar tensor of shape [R, A, E, 6] as input,
      where each grid point combines Doppler velocity and intensity from the
      current and previous two frames to form 4D spatiotemporal features."

Layout canali del tensore output [R, A, E, 6]:
    canali [0, 1] → frame t   (corrente):  [intensità_t,   Doppler_t]
    canali [2, 3] → frame t-1 (precedente): [intensità_t-1, Doppler_t-1]
    canali [4, 5] → frame t-2 (più vecchio): [intensità_t-2, Doppler_t-2]

Nota: il primo e il secondo frame di una sequenza non hanno abbastanza
storia → usa padding con zeri (o frame duplicati — scegli e documenta).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class TemporalFusion:
    """
    Buffer circolare per la fusione temporale di 3 frame radar.

    Mantiene in memoria i tensori [R, A, E, 2] degli ultimi N frame
    e li combina in [R, A, E, 6] quando richiesto.
    """

    def __init__(
        self,
        n_frames: int = 3,
        tensor_shape: tuple[int, int, int] = (480, 736, 11),
        padding_mode: str = "zeros",
    ) -> None:
        """
        Args:
            n_frames:     numero di frame da fondere (default: 3).
            tensor_shape: shape (R, A, E) del singolo frame.
            padding_mode: come gestire i frame mancanti all'inizio della sequenza.
                          "zeros"      → padding con tensori a zero.
                          "replicate"  → replica il primo frame disponibile.
        """
        assert padding_mode in ("zeros", "replicate"), \
            f"padding_mode deve essere 'zeros' o 'replicate', ricevuto: {padding_mode}"

        self.n_frames = n_frames
        self.tensor_shape = tensor_shape  # (R, A, E)
        self.padding_mode = padding_mode

        # Buffer degli ultimi n_frames tensori [R, A, E, 2]
        self._buffer: deque[np.ndarray] = deque(maxlen=n_frames)

    @property
    def channels_per_frame(self) -> int:
        """Canali per singolo frame (intensità + Doppler = 2)."""
        return 2

    @property
    def total_channels(self) -> int:
        """Canali totali nel tensore fuso = n_frames × 2."""
        return self.n_frames * self.channels_per_frame

    def reset(self) -> None:
        """Svuota il buffer — da chiamare all'inizio di ogni sequenza."""
        self._buffer.clear()

    def push(self, tensor_rae2: np.ndarray) -> np.ndarray:
        """
        Aggiunge un nuovo frame al buffer e restituisce il tensore fuso.

        Args:
            tensor_rae2: np.ndarray, shape (R, A, E, 2) — frame corrente.

        Returns:
            fused: np.ndarray, shape (R, A, E, 6) — tensore fuso pronto per il modello.
        """
        assert tensor_rae2.shape == (*self.tensor_shape, self.channels_per_frame), \
            f"Shape attesa {(*self.tensor_shape, self.channels_per_frame)}, ricevuta {tensor_rae2.shape}"

        self._buffer.appendleft(tensor_rae2)  # il più recente è a sinistra
        return self.get_fused()

    def get_fused(self) -> np.ndarray:
        """
        Costruisce il tensore fuso [R, A, E, 6] dal buffer corrente.

        I frame mancanti (all'inizio della sequenza) sono gestiti con padding.

        Returns:
            fused: np.ndarray, shape (R, A, E, n_frames * 2).
        """
        frames = list(self._buffer)   # frame[0] = più recente, frame[-1] = più vecchio

        # Padding se non abbiamo ancora abbastanza frame
        while len(frames) < self.n_frames:
            if self.padding_mode == "zeros":
                pad = np.zeros((*self.tensor_shape, self.channels_per_frame), dtype=np.float32)
            else:  # "replicate" — replica l'ultimo frame disponibile
                pad = frames[-1] if frames else np.zeros(
                    (*self.tensor_shape, self.channels_per_frame), dtype=np.float32
                )
            frames.append(pad)

        # Assicurati di avere esattamente n_frames frame
        frames = frames[:self.n_frames]

        # Stack lungo l'asse dei canali: (R, A, E, 2) × 3 → (R, A, E, 6)
        fused = np.concatenate(frames, axis=-1).astype(np.float32)

        assert fused.shape == (*self.tensor_shape, self.total_channels)
        return fused


# ------------------------------------------------------------------
# Funzione stateless per usare direttamente 3 tensori già disponibili
# ------------------------------------------------------------------

def fuse_three_frames(
    tensor_t:    np.ndarray,
    tensor_tm1:  Optional[np.ndarray] = None,
    tensor_tm2:  Optional[np.ndarray] = None,
    padding_mode: str = "zeros",
) -> np.ndarray:
    """
    Fonde 3 tensori [R, A, E, 2] in [R, A, E, 6].

    Utile quando si carica direttamente una tripla di frame senza
    processare una sequenza in ordine cronologico.

    Args:
        tensor_t:   frame corrente t (obbligatorio).
        tensor_tm1: frame t-1 (opzionale — padding se None).
        tensor_tm2: frame t-2 (opzionale — padding se None).
        padding_mode: "zeros" o "replicate".

    Returns:
        fused: np.ndarray, shape (R, A, E, 6).
    """
    R, A, E, C = tensor_t.shape
    assert C == 2, f"Ogni frame deve avere 2 canali, ricevuti: {C}"

    zero_frame = np.zeros((R, A, E, 2), dtype=np.float32)

    if tensor_tm1 is None:
        tensor_tm1 = zero_frame if padding_mode == "zeros" else tensor_t.copy()
    if tensor_tm2 is None:
        tensor_tm2 = zero_frame if padding_mode == "zeros" else tensor_t.copy()

    # Ordine: [t, t-1, t-2] lungo l'asse canali
    fused = np.concatenate([tensor_t, tensor_tm1, tensor_tm2], axis=-1).astype(np.float32)
    return fused


# ------------------------------------------------------------------
# Normalizzazione del tensore (scelta da documentare nel paper)
# ------------------------------------------------------------------

def normalize_radar_tensor(
    tensor: np.ndarray,
    intensity_log: bool = True,
    velocity_normalize: bool = True,
    velocity_max_mps: float = 20.0,
) -> np.ndarray:
    """
    Normalizza i canali del tensore radar prima di darli in input al modello.

    Il paper non specifica esplicitamente la normalizzazione — questa è
    una scelta implementativa che va documentata e potenzialmente ablata.

    Args:
        tensor:            np.ndarray, shape (..., 6) — canali [I0, D0, I1, D1, I2, D2].
        intensity_log:     se True, applica log1p all'intensità (comprime la dinamica).
        velocity_normalize: se True, normalizza Doppler in [-1, 1].
        velocity_max_mps:  velocità massima attesa per normalizzazione.

    Returns:
        normalized: np.ndarray, stessa shape, float32.

    TODO:
        Considera se usare normalizzazione globale (statistiche sul training set)
        o per-frame. La normalizzazione globale è più stabile ma richiede
        di calcolare media/std sull'intero dataset.
    """
    t = tensor.astype(np.float64).copy()

    # Canali intensità: indici 0, 2, 4
    intensity_channels = [0, 2, 4]
    # Canali Doppler: indici 1, 3, 5
    doppler_channels   = [1, 3, 5]

    if intensity_log:
        for c in intensity_channels:
            t[..., c] = np.log1p(t[..., c])
        # Normalizza in [0, 1] dividendo per il massimo del frame
        for c in intensity_channels:
            max_val = t[..., c].max()
            if max_val > 0:
                t[..., c] /= max_val

    if velocity_normalize:
        for c in doppler_channels:
            t[..., c] = np.clip(t[..., c] / velocity_max_mps, -1.0, 1.0)

    return t.astype(np.float32)
