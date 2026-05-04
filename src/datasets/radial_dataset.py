"""
RADIal Dataset — PyTorch Dataset
==================================
Carica i file .npz pre-processati da scripts/preprocess_radial.py.

════════════════════════════════════════════════════════════════
WORKFLOW (eseguire nell'ordine)
════════════════════════════════════════════════════════════════
1. Preprocessing offline (UNA SOLA VOLTA):
       python scripts/preprocess_radial.py --config configs/radial.yaml --workers 8
   → legge fft_{id}.npy (512,256,16) complex + CalibrationTable.npy
   → applica MIMO assembly + AoA beamforming 3D
   → salva processed/{id:06d}.npz con:
       tensor_rae2  (480, 736, 11, 2)  float32   [intensity, velocity]
       rad_map      (480, 736, 16)     float32   [RAD map Doppler ridotto]
       lidar_occ    (480, 736, 11)     float32   [GT occupancy binaria]
   → salva processed/index.json (sample_id → split)

2. Training:
       python scripts/train.py --config configs/radial.yaml
   → carica i .npz direttamente (veloce, niente AoA a runtime)

════════════════════════════════════════════════════════════════
Struttura del dataset RADIal su server:
    /media/data/matteo-carnevale/dataset/RADIal/
    ├── labels.csv                           ← split + metadata
    ├── radar_FFT/fft_{:06d}.npy            ← (512,256,16) complex RD spectrums
    ├── laser_PCL/laser_PCL_{:06d}.npy      ← LiDAR Velodyne strutturato
    └── CalibrationTable.npy                 ← per AoA (SignalProcessing/)

    /media/data/matteo-carnevale/dataset/RADIal_processed/
    ├── index.json
    ├── 000042.npz
    ...

════════════════════════════════════════════════════════════════
Griglia RADIal (paper Section 4.1):
    R=480, A=736, E=11
    range [0,50]m, azimuth [-75°,75°], elevation [-4°,6°]
    D=16 (Doppler bins ridotti nella RAD map)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Griglia RADIal (paper Section 4.1)
GRID_R, GRID_A, GRID_E = 480, 736, 11
RAD_D = 16   # Doppler bins ridotti nella RAD map (N_CHIRPS/N_RED_DOP = 256/16)

# Split test ufficiale (loader.py del repo RADIal)
_TEST_SEQS = {
    "RECORD@2020-11-22_12.45.05", "RECORD@2020-11-22_12.25.47",
    "RECORD@2020-11-22_12.03.47", "RECORD@2020-11-22_12.54.38",
    "RECORD@2020-11-22_12.49.56", "RECORD@2020-11-22_12.11.49",
    "RECORD@2020-11-22_12.28.47", "RECORD@2020-11-21_14.25.06",
}


class RADIalDataset(Dataset):
    """
    Dataset RADIal per Radar-Mamba.

    Carica triplet di frame consecutivi da file .npz pre-processati.
    Ogni campione restituisce:
        radar_cube  (6, R=480, A=736, E=11)  ← 3 frame × [intensity, velocity]
        rad_map     (D=16, R=480, A=736)     ← RAD map del frame t
        lidar_occ   (R=480, A=736, E=11)     ← GT occupancy
    """

    def __init__(
        self,
        processed_path: str | Path,
        mode:     str  = "train",   # "train" | "test"
        n_frames: int  = 3,
        transform=None,
    ) -> None:
        """
        Args:
            processed_path: directory contenente index.json e i file .npz
                            (output di scripts/preprocess_radial.py).
                            Per default: /media/data/matteo-carnevale/dataset/RADIal_processed
            mode:           "train" o "test".
            n_frames:       numero di frame consecutivi per la fusione temporale (3).
        """
        assert mode in ("train", "test"), f"mode deve essere 'train' o 'test', ricevuto: {mode}"

        self.proc_path = Path(processed_path)
        self.mode      = mode
        self.n_frames  = n_frames
        self.transform = transform

        # Carica indice
        index_path = self.proc_path / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json non trovato in {self.proc_path}\n"
                "Esegui prima il preprocessing:\n"
                "  python scripts/preprocess_radial.py --config configs/radial.yaml"
            )
        with open(index_path) as f:
            all_entries = json.load(f)

        # Filtra per split e verifica che il .npz esiste
        entries = [
            e for e in all_entries
            if e["split"] == mode and Path(e["npz_path"]).exists()
        ]
        if not entries:
            raise ValueError(
                f"Nessun campione trovato per split='{mode}' in {self.proc_path}\n"
                f"Totale nell'indice: {len(all_entries)}"
            )

        # Ordina per sample_id e costruisci le triplet [t-2, t-1, t]
        entries.sort(key=lambda x: x["sample_id"])
        self.all_entries = entries
        self.samples     = self._build_triplets(entries)

        print(f"RADIalDataset [{mode}]: {len(self.samples)} triplet "
              f"da {len(entries)} frame")

    # ------------------------------------------------------------------
    # Costruzione triplet
    # ------------------------------------------------------------------

    def _build_triplets(self, entries: list[dict]) -> list[list[dict]]:
        """
        Raggruppa i frame in triplet [t-2, t-1, t].

        Per i frame vicini all'inizio di una sequenza, usa padding
        replicando il frame più vecchio disponibile.
        """
        triplets = []
        for i, entry in enumerate(entries):
            t   = entry
            tm1 = entries[max(0, i - 1)]
            tm2 = entries[max(0, i - 2)]
            triplets.append([tm2, tm1, t])
        return triplets

    # ------------------------------------------------------------------
    # Caricamento .npz
    # ------------------------------------------------------------------

    def _load_npz(self, entry: dict) -> dict:
        """Carica un singolo frame da .npz."""
        data = np.load(entry["npz_path"])
        return {
            "tensor_rae2": data["tensor_rae2"],   # (480, 736, 11, 2) float32
            "rad_map":     data["rad_map"],        # (480, 736, 16)    float32
            "lidar_occ":   data["lidar_occ"],      # (480, 736, 11)    float32
        }

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        tm2_info, tm1_info, t_info = self.samples[idx]

        # Carica i 3 frame
        t   = self._load_npz(t_info)
        tm1 = self._load_npz(tm1_info)
        tm2 = self._load_npz(tm2_info)

        # Fusione temporale: [t, t-1, t-2] lungo asse canali → (R,A,E,6)
        # Canali [0,1]=t, [2,3]=t-1, [4,5]=t-2
        radar_cube = np.concatenate(
            [t["tensor_rae2"], tm1["tensor_rae2"], tm2["tensor_rae2"]],
            axis=-1
        )   # (480, 736, 11, 6)

        # Converti in tensori PyTorch channel-first
        sample = {
            "radar_cube": torch.from_numpy(
                radar_cube.transpose(3, 0, 1, 2)   # (6, R, A, E)
            ),
            "rad_map": torch.from_numpy(
                t["rad_map"].transpose(2, 0, 1)     # (D, R, A)
            ),
            "lidar_occ": torch.from_numpy(t["lidar_occ"]),   # (R, A, E)
            "meta": {
                "sample_id": t_info["sample_id"],
                "split":     self.mode,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Proprietà utili
    # ------------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Forma della griglia (R, A, E)."""
        return (GRID_R, GRID_A, GRID_E)

    @property
    def n_doppler_bins(self) -> int:
        """Numero di bin Doppler nella RAD map."""
        return RAD_D
