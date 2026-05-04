"""
Radar-Mamba U-Net — modello completo
======================================
REF: Paper Section 3.3.2 + Fig. 1(a2) — 2.4M parametri

Pipeline:
  radar_cube (B,6,R,A,E) → EDCEncoder → [B,64,R,A]
                            ↓ RM Block ×N (encoder)
                            ↓ CBAM (bottleneck)
                            ↓ DEF Block ← DopplerBackbone ← rad_map (B,D,R,A)
                            ↓ Decoder → logits (B,R,A,E)

Nota architetturale:
    Il DEF Block è posizionato dopo il bottleneck CBAM, prima del decoder.
    Questo è il punto in cui le feature di elevazione-tempo (dall'encoder main)
    vengono arricchite con le feature di velocità Doppler (dal backbone RAD).
    È consistente con Fig. 1(a3) che mostra i due rami convergere nel DEF
    prima di produrre il dense point cloud finale.

Post-processing inferenza:
    1. occupancy = sigmoid(logits) > threshold
    2. Indici voxel True → coordinate polari (r_m, az_deg, el_deg)
    3. (Opzionale) Aggiunge Doppler I dal tensore radar originale per
       produrre il punto cloud completo [R, A, E, D, I]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from src.models.edc_encoder    import EDCEncoder
from src.models.rm_block       import RMBlock
from src.models.cbam           import CBAM
from src.models.def_block      import DEFBlock
from src.models.doppler_backbone import DopplerBackbone
from src.models.decoder        import OccupancyDecoder


class RadarMambaUNet(nn.Module):
    """
    Radar-Mamba U-Net completo — 2.4M parametri (paper).

    I/O:
        radar_cube: (B, 6, R, A, E)  — radar tensor fuso temporalmente
        rad_map:    (B, D, R, A)     — RAD map (Doppler dimension first)
        → logits:   (B, R, A, E)     — occupancy logits (prima del sigmoid)
    """

    def __init__(
        self,
        radar_channels:   int = 6,
        doppler_bins:     int = 128,    # D — Doppler bins RAD map
        base_channels:    int = 64,
        n_elevation_bins: int = 11,     # E — varia per dataset (11 o 34)
        n_encoder_blocks: int = 3,      # RM Blocks nell'encoder
        n_doppler_blocks: int = 2,      # RM Blocks nel Doppler backbone
        cbam_reduction:   int = 16,
        rm_kwargs:        dict | None = None,
    ) -> None:
        super().__init__()
        if rm_kwargs is None:
            rm_kwargs = {}

        C = base_channels   # 64

        # ── 1. EDC Encoder: (B,6,R,A,E) → (B,64,R,A) ────────────────
        self.edc = EDCEncoder(in_channels=radar_channels, out_channels=C)

        # ── 2. Encoder stack: N × RM Block (full resolution BEV) ─────
        self.encoder_blocks = nn.ModuleList([
            RMBlock(in_channels=C, **rm_kwargs)
            for _ in range(n_encoder_blocks)
        ])

        # ── 3. CBAM bottleneck ────────────────────────────────────────
        self.cbam = CBAM(in_channels=C, reduction_ratio=cbam_reduction)

        # ── 4. Doppler backbone: (B,D,R,A) → (B,64,R,A) ─────────────
        self.doppler_backbone = DopplerBackbone(
            in_channels=doppler_bins,
            out_channels=C,
            n_blocks=n_doppler_blocks,
            rm_kwargs=rm_kwargs,
        )

        # ── 5. DEF Block: fonde Doppler + bottleneck (elevation) ─────
        self.def_block = DEFBlock(
            doppler_ch=C,
            elevation_ch=C,
            hidden_ch=C,
            out_ch=C,
        )

        # ── 6. Decoder: (B,64,R,A) → logits (B,R,A,E) ────────────────
        self.decoder = OccupancyDecoder(
            in_channels=C,
            n_elevation_bins=n_elevation_bins,
            mid_channels=[C, C // 2],
            use_skip=False,   # semplice per rispettare budget parametri
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        radar_cube: torch.Tensor,   # (B, 6, R, A, E)
        rad_map:    torch.Tensor,   # (B, D, R, A)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, R, A, E) — occupancy logits (prima del sigmoid).
                    Applica torch.sigmoid(logits) per avere probabilità ∈ [0,1].
        """
        # 1. EDC: comprime (B,6,R,A,E) → (B,64,R,A)
        bev = self.edc(radar_cube)

        # 2. Encoder RM Blocks
        for block in self.encoder_blocks:
            bev = block(bev)

        # 3. CBAM bottleneck
        bev = self.cbam(bev)

        # 4. Doppler backbone (ramo parallelo)
        dop_feat = self.doppler_backbone(rad_map)   # (B, 64, R, A)

        # 5. DEF Block: fondi Doppler + bottleneck elevation features
        bev = self.def_block(
            doppler_feat=dop_feat,
            elevation_feat=bev,
        )   # (B, 64, R, A)

        # 6. Decoder: (B,64,R,A) → (B,R,A,E) logits
        logits = self.decoder(bev)

        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        radar_cube: torch.Tensor,
        rad_map:    torch.Tensor,
        threshold:  float = 0.5,
    ) -> torch.Tensor:
        """
        Predice l'occupancy binaria.

        Returns:
            occupancy: (B, R, A, E) bool.
        """
        logits = self.forward(radar_cube, rad_map)
        return torch.sigmoid(logits) > threshold

    @torch.no_grad()
    def predict_pointcloud(
        self,
        radar_cube: torch.Tensor,    # (B, 6, R, A, E)
        rad_map:    torch.Tensor,    # (B, D, R, A)
        range_axis:     np.ndarray,  # (R,) in metri
        azimuth_axis:   np.ndarray,  # (A,) in radianti
        elevation_axis: np.ndarray,  # (E,) in radianti
        threshold: float = 0.5,
    ) -> list[np.ndarray]:
        """
        Converte i logits in una lista di point cloud 3D in coordinate polari.

        Args:
            range/azimuth/elevation_axis: assi fisici della griglia.
            threshold: soglia occupancy.

        Returns:
            List[ndarray] — per ogni sample nel batch, shape (N_i, 3):
                colonne [range_m, azimuth_rad, elevation_rad].
        """
        occupancy = self.predict(radar_cube, rad_map, threshold)   # (B, R, A, E)
        batch_pcs = []

        for b in range(occupancy.shape[0]):
            occ_b = occupancy[b].cpu().numpy()         # (R, A, E)
            idx_r, idx_a, idx_e = np.where(occ_b)     # indici dei voxel occupati

            if len(idx_r) == 0:
                batch_pcs.append(np.zeros((0, 3), dtype=np.float32))
                continue

            # Converti indici → valori fisici interpolando sugli assi
            r_vals  = range_axis[np.clip(idx_r, 0, len(range_axis) - 1)]
            az_vals = azimuth_axis[np.clip(idx_a, 0, len(azimuth_axis) - 1)]
            el_vals = elevation_axis[np.clip(idx_e, 0, len(elevation_axis) - 1)]

            pts = np.stack([r_vals, az_vals, el_vals], axis=-1).astype(np.float32)
            batch_pcs.append(pts)

        return batch_pcs

    def count_parameters(self) -> int:
        """Conta i parametri trainabili (obiettivo: ~2.4M)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------
# Factory per dataset
# ------------------------------------------------------------------

def build_model_for_radial() -> RadarMambaUNet:
    """
    Costruisce il modello con parametri per RADIal.
    Grid: [R=480, A=736, E=11], RAD map: D=64 (doppler bins disponibili).
    """
    return RadarMambaUNet(
        radar_channels=6,
        doppler_bins=64,
        base_channels=64,
        n_elevation_bins=11,
        n_encoder_blocks=3,
        n_doppler_blocks=2,
    )


def build_model_for_radelft() -> RadarMambaUNet:
    """
    Costruisce il modello con parametri per RaDelft.

    Grid dati reali: [R=500, A=240, E=34]
    (paper usa [512, 256, 34] con zero-padding — equivalente)
    RAD map: D=128 Doppler bins (vel_fft_size da data_preparation.py)
    """
    return RadarMambaUNet(
        radar_channels=6,
        doppler_bins=128,    # D=128 = vel_fft_size di RaDelft
        base_channels=64,
        n_elevation_bins=34, # E=34 bin fisici di elevation
        n_encoder_blocks=3,
        n_doppler_blocks=2,
    )
