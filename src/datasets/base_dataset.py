"""
Dataset Base — contratto comune per RADIal e RaDelft
======================================================
Definisce l'interfaccia che entrambi i dataset devono rispettare.
Il modello consuma sempre lo stesso dizionario di tensori, indipendentemente
dal dataset sottostante.

Contratto __getitem__:
    {
        "radar_cube":  torch.Tensor (6, R, A, E)  float32  — input U-Net
                       (canali first: trasposto da (R,A,E,6) per PyTorch)
        "rad_map":     torch.Tensor (D, R, A)     float32  — input Doppler backbone
                       (dim first: trasposto da (R,A,D))
        "lidar_occ":   torch.Tensor (R, A, E)     float32  — ground truth binary
        "meta":        dict  — scene_id, frame_idx, timestamp (non tensorizzato)
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class RadarMambaDataset(ABC, Dataset):
    """
    Dataset base astratto per Radar-Mamba.

    Subclass: RADIalDataset, RaDelftDataset.
    """

    def __init__(
        self,
        index_file: str | Path,
        cfg: Any,
        split: str = "train",
        transform=None,
    ) -> None:
        """
        Args:
            index_file: path al JSON/CSV con la lista dei campioni
                        (generato da scripts/build_dataset_index.py).
            cfg:        OmegaConf DictConfig del dataset (es. radial.yaml).
            split:      "train" | "test" | "val".
            transform:  trasformazioni da applicare al campione (es. data augmentation).
        """
        self.index_file = Path(index_file)
        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.samples = self._load_index()

    # ------------------------------------------------------------------
    # Metodi astratti — da implementare nelle subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_index(self) -> list[dict]:
        """
        Carica l'indice dei campioni dal file JSON/CSV.

        Returns:
            lista di dict, uno per campione. Ogni dict deve contenere almeno:
            {
                "scene_id":   str   — identificatore della scena/sequenza
                "frame_idx":  int   — indice del frame corrente t
                "frame_paths": list[str] — path ai frame [t-2, t-1, t]
                "lidar_path": str   — path al point cloud LiDAR per il frame t
                "split":      str   — "train" | "test"
            }
        """
        ...

    @abstractmethod
    def _load_sample(self, sample_info: dict) -> dict:
        """
        Carica e preprocessa un campione dall'indice.

        Args:
            sample_info: dizionario dell'indice (output di _load_index).

        Returns:
            dict con:
                "radar_cube": np.ndarray (R, A, E, 6)
                "rad_map":    np.ndarray (R, A, D)
                "lidar_occ":  np.ndarray (R, A, E)
                "meta":       dict
        """
        ...

    # ------------------------------------------------------------------
    # Metodi comuni (non richiedono override)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        sample = self._load_sample(sample_info)

        if self.transform is not None:
            sample = self.transform(sample)

        return self._to_tensors(sample)

    @staticmethod
    def _to_tensors(sample: dict) -> dict:
        """
        Converte i numpy array in torch.Tensor e traspone nel formato
        "channels first" atteso da PyTorch.
        """
        radar_cube = torch.from_numpy(
            sample["radar_cube"].transpose(3, 0, 1, 2)  # (R,A,E,6) → (6,R,A,E)
        )
        rad_map = torch.from_numpy(
            sample["rad_map"].transpose(2, 0, 1)         # (R,A,D) → (D,R,A)
        )
        lidar_occ = torch.from_numpy(sample["lidar_occ"])   # (R,A,E) → rimane (R,A,E)

        return {
            "radar_cube": radar_cube,   # (6, R, A, E)
            "rad_map":    rad_map,      # (D, R, A)
            "lidar_occ":  lidar_occ,    # (R, A, E)
            "meta":       sample["meta"],
        }

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Dimensioni (R, A, E) della griglia dal config."""
        g = self.cfg.dataset.grid
        return (g.R, g.A, g.E)
