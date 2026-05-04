"""
Radar Cube Builder — orchestratore ADC → tensore [R,A,E,6] + RAD map
======================================================================
Coordina ADCProcessor + DoAEstimator + TemporalFusion per produrre
i due input del modello da un gruppo di 3 frame ADC consecutivi.

Output per un campione di training:
    radar_cube: np.ndarray (R, A, E, 6)  → input alla U-Net
    rad_map:    np.ndarray (R, A, D)     → input al Doppler backbone
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.radar_tensor.adc_processing import ADCProcessor
from src.radar_tensor.doa_estimation import DoAEstimator
from src.radar_tensor.temporal_fusion import TemporalFusion, fuse_three_frames


class RadarCubeBuilder:
    """
    Costruisce il cubo radar [R,A,E,6] e la mappa RAD [R,A,D] da tre frame ADC.

    Uso tipico (offline — si passa una tripla di path):
        builder = RadarCubeBuilder.for_radial(radar_cfg, grid_cfg)
        result  = builder.build_from_paths(paths=[path_t2, path_t1, path_t])
        cube    = result["radar_cube"]    # (R, A, E, 6)
        rad     = result["rad_map"]       # (R, A, D)
    """

    def __init__(
        self,
        adc_processor: ADCProcessor,
        doa_estimator: DoAEstimator,
        dataset_name: str = "radial",
        normalize: bool = True,
    ) -> None:
        self.adc_processor = adc_processor
        self.doa_estimator = doa_estimator
        self.dataset_name  = dataset_name
        self.normalize     = normalize

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_radial(cls, radar_cfg: dict, grid_cfg: dict) -> "RadarCubeBuilder":
        """Crea un builder configurato per il dataset RADIal."""
        return cls(
            adc_processor = ADCProcessor(radar_cfg),
            doa_estimator = DoAEstimator(radar_cfg, grid_cfg),
            dataset_name  = "radial",
        )

    @classmethod
    def for_radelft(cls, radar_cfg: dict, grid_cfg: dict) -> "RadarCubeBuilder":
        """Crea un builder configurato per il dataset RaDelft."""
        return cls(
            adc_processor = ADCProcessor(radar_cfg),
            doa_estimator = DoAEstimator(radar_cfg, grid_cfg),
            dataset_name  = "radelft",
        )

    # ------------------------------------------------------------------
    # Processing di un singolo frame
    # ------------------------------------------------------------------

    def process_single_frame(self, frame_path: str | Path) -> dict:
        """
        Processa un singolo frame ADC → tensore [R,A,E,2] + RAD map.

        Args:
            frame_path: path al file/cartella ADC del frame.

        Returns:
            dict con:
                "tensor_rae2": np.ndarray (R, A, E, 2)
                "rad_map":     np.ndarray (R, A, D)

        TODO:
            1. Carica ADC con il metodo corretto per il dataset:
                   if self.dataset_name == "radial":
                       adc = ADCProcessor.load_radial_adc(frame_path)
                   elif self.dataset_name == "radelft":
                       adc = ADCProcessor.load_radelft_adc(frame_path)

            2. Calcola la Range-Doppler map:
                   virtual_array = self.adc_processor.compute_rd_map(adc)
                   # o il metodo che produca il virtual array MIMO

            3. Stima DoA:
                   doa_result = self.doa_estimator.estimate(virtual_array)

            4. Ritorna doa_result (che contiene già tensor_rae2 e rad_map).
        """
        raise NotImplementedError(
            "TODO: carica ADC, calcola virtual array MIMO, stima DoA."
        )

    # ------------------------------------------------------------------
    # Fusione temporale di 3 frame
    # ------------------------------------------------------------------

    def build_from_paths(
        self,
        paths: Sequence[str | Path],
        padding_mode: str = "zeros",
    ) -> dict:
        """
        Costruisce il cubo radar [R,A,E,6] da una tripla di frame.

        Args:
            paths: lista di path ai frame [path_t-2, path_t-1, path_t].
                   Deve contenere 1, 2 o 3 path (padding per quelli mancanti).
            padding_mode: "zeros" o "replicate".

        Returns:
            dict con:
                "radar_cube": np.ndarray (R, A, E, 6)  — input alla U-Net
                "rad_map":    np.ndarray (R, A, D)     — input al backbone Doppler
                              (usiamo la RAD map del frame più recente)
        """
        assert 1 <= len(paths) <= 3, "Fornire 1-3 path per la fusione temporale."

        # Processa ogni frame disponibile
        results = [self.process_single_frame(p) for p in paths]

        # Estrai tensori [R,A,E,2] per ogni frame disponibile
        tensors = [r["tensor_rae2"] for r in results]

        # Il frame più recente è l'ultimo nella lista
        tensor_t   = tensors[-1]
        tensor_tm1 = tensors[-2] if len(tensors) >= 2 else None
        tensor_tm2 = tensors[-3] if len(tensors) >= 3 else None

        radar_cube = fuse_three_frames(
            tensor_t, tensor_tm1, tensor_tm2, padding_mode=padding_mode
        )

        if self.normalize:
            from src.radar_tensor.temporal_fusion import normalize_radar_tensor
            radar_cube = normalize_radar_tensor(radar_cube)

        # RAD map dal frame più recente
        rad_map = results[-1]["rad_map"]

        return {
            "radar_cube": radar_cube,   # (R, A, E, 6)
            "rad_map":    rad_map,      # (R, A, D)
        }

    def build_from_tensors(
        self,
        tensor_t:   np.ndarray,
        tensor_tm1: np.ndarray | None = None,
        tensor_tm2: np.ndarray | None = None,
        rad_map:    np.ndarray | None = None,
        padding_mode: str = "zeros",
    ) -> dict:
        """
        Fusione temporale da tensori [R,A,E,2] già pre-calcolati.
        Utile quando i tensori sono salvati su disco (modalità cached).

        Args:
            tensor_t, tensor_tm1, tensor_tm2: tensori singolo frame, shape (R,A,E,2).
            rad_map: RAD map del frame t, shape (R,A,D). Se None viene lasciata None.
            padding_mode: gestione frame mancanti.

        Returns:
            dict con "radar_cube" e "rad_map".
        """
        radar_cube = fuse_three_frames(tensor_t, tensor_tm1, tensor_tm2, padding_mode)

        if self.normalize:
            from src.radar_tensor.temporal_fusion import normalize_radar_tensor
            radar_cube = normalize_radar_tensor(radar_cube)

        return {"radar_cube": radar_cube, "rad_map": rad_map}
