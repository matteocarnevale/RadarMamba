"""
RADIal Dataset — PyTorch Dataset
==================================
REF: https://github.com/valeoai/RADIal
     FFTRadNet/dataset/dataset.py + loader/loader.py + SignalProcessing/rpl.py

Struttura attesa sul disco (dalla repo RADIal):
    data/raw/radial/
    ├── labels.csv                              ← bounding box + metadata
    ├── radar_FFT/
    │   └── fft_{:06d}.npy                     ← Range-Doppler FFT complessa (R×D×C)
    ├── laser_PCL/                              ← LiDAR point cloud per ogni sample
    │   └── laser_PCL_{:06d}.npy               ← (N, 6) Velodyne structured array
    └── camera/
        └── image_{:06d}.jpg

Il dataset "ready-to-use" di RADIal ha già:
  - `radar_FFT/fft_{:06d}.npy`: FFT complessa range×doppler×antenne
  - I frame CONSECUTIVI NON sono espliciti → usiamo sample_id ± 1, ± 2

Per Radar-Mamba:
  - Costruiamo il tensore [R, A, E, 6] dalla FFT complessa usando la
    CalibrationTable.npy del repo (AoA beamforming)
  - In mancanza della calibration table (modalità "fft_only"), usiamo
    la FFT come proxy flat della RAD map e un tensore [R, A, E=1, 2] semplificato

Split ufficiale (da loader.py):
    Validation: RECORD@2020-11-22_12.49.56, ..., RECORD@2020-11-22_12.11.49,
                RECORD@2020-11-22_12.28.47, RECORD@2020-11-21_14.25.06
    Test:       RECORD@2020-11-22_12.45.05, ..., altri 3

Per Radar-Mamba paper, viene usato uno split sequenze 81 train / 10 test.
Usiamo lo split per sequenza come nel paper (non per sample_id come nell'originale).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.alignment.voxelization import Voxelizer

# Sequenze di test da loader.py (4 val + 4 test = 8 riservate)
_TEST_SEQS = {
    "RECORD@2020-11-22_12.45.05",
    "RECORD@2020-11-22_12.25.47",
    "RECORD@2020-11-22_12.03.47",
    "RECORD@2020-11-22_12.54.38",
    "RECORD@2020-11-22_12.49.56",
    "RECORD@2020-11-22_12.11.49",
    "RECORD@2020-11-22_12.28.47",
    "RECORD@2020-11-21_14.25.06",
}

# Griglia RADIal (paper Section 4.1)
_R, _A, _E = 480, 736, 11

# Parametri fisici griglia RADIal
_RANGE_MAX_M     = 50.0
_AZ_MIN_DEG      = -75.0
_AZ_MAX_DEG      =  75.0
_EL_MIN_DEG      = -4.0
_EL_MAX_DEG      =  6.0

# Parametri radar RADIal (da rpl.py)
_N_SAMPLES_PER_CHIRP = 512
_N_CHIRPS            = 256
_N_RX                = 16
_N_TX                = 12
# Range: range = range_bin / n_samples * 103.0 (da rpl.py __get_PCL)
_RANGE_SCALE         = 103.0   # m (range massimo del chirp)


class RADIalDataset(Dataset):
    """
    Dataset RADIal per Radar-Mamba.

    Due modalità:
      "fft_only":  usa direttamente fft_{:06d}.npy (più veloce, nessuna calibration)
                   La FFT ha shape (R_fft=512, D=256, 16_rx) → usiamo [512, 16, 2] per
                   intensità e Doppler, poi risampling a [R=480, A=736, E=11, 6].
                   ATTENZIONE: senza la CalibrationTable non c'è vera stima DoA.
                   Usare questo solo per debug/sviluppo.

      "calibrated": usa rpl.RadarSignalProcessing per la stima completa AoA.
                    Richiede CalibrationTable.npy e mkl_fft (solo su Linux con Intel MKL).
                    Produce il tensore [R, A, E, 2] fedele al paper.
    """

    def __init__(
        self,
        root_dir:         str | Path,
        mode:             str = "train",
        processing_mode:  str = "fft_only",   # "fft_only" | "calibrated"
        calib_table_path: str | None = None,
        normalize:        bool = True,
        transform=None,
    ) -> None:
        """
        Args:
            root_dir:         path alla root del dataset ready-to-use di RADIal.
            mode:             "train" | "test".
            processing_mode:  modalità di processing.
            calib_table_path: path a CalibrationTable.npy (solo per "calibrated").
            normalize:        applica normalizzazione ai tensori.
        """
        self.root    = Path(root_dir)
        self.mode    = mode
        self.proc    = processing_mode
        self.norm    = normalize
        self.transform = transform

        # Voxelizer per LiDAR
        self.voxelizer = Voxelizer.for_radial()

        # Leggi CSV labels (REF: FFTRadNet/dataset/dataset.py)
        labels_path = self.root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv non trovato in {self.root}")
        self.labels_df = pd.read_csv(labels_path)

        # Colonna dataset = nome della sequenza
        # Filtra per split: sample_id univoci nella sequenza corretta
        if "dataset" in self.labels_df.columns:
            is_test = self.labels_df["dataset"].isin(_TEST_SEQS)
            if mode == "test":
                df = self.labels_df[is_test]
            else:
                df = self.labels_df[~is_test]
        else:
            # Fallback: usa tutti i sample_id (ignora split sequenze)
            df = self.labels_df

        # sample_id univoci
        self.sample_ids = sorted(df.iloc[:, 0].unique().tolist())

        # Modalità calibrated: inizializza RadarSignalProcessing
        self._rsp = None
        if processing_mode == "calibrated":
            self._init_signal_processing(calib_table_path)

    # ------------------------------------------------------------------

    def _init_signal_processing(self, calib_path: str | None) -> None:
        """
        Inizializza RadarSignalProcessing dal repo RADIal.
        REF: SignalProcessing/rpl.py::RadarSignalProcessing
        """
        if calib_path is None:
            # Cerca CalibrationTable.npy nella SignalProcessing/ del repo
            default = self.root.parent / "SignalProcessing" / "CalibrationTable.npy"
            if default.exists():
                calib_path = str(default)
            else:
                raise FileNotFoundError(
                    "CalibrationTable.npy non trovato. Specifica calib_table_path "
                    "o scarica il repo RADIal (contiene SignalProcessing/CalibrationTable.npy)."
                )

        try:
            import sys
            # Il modulo rpl.py del repo RADIal deve essere nel PYTHONPATH
            from rpl import RadarSignalProcessing as RSP
            # Modalità RA: usa una elevazione (quella parallela alla strada, idx=5)
            # Per la stima completa 3D useremmo method='PC' ma è più pesante
            self._rsp = RSP(path_calib_mat=calib_path, method="RA", device="cpu", lib="PyTorch")
        except ImportError:
            raise ImportError(
                "Impossibile importare rpl.py. Aggiungi il path al repo RADIal:\n"
                "  sys.path.insert(0, 'path/to/RADIal/SignalProcessing')\n"
                "  from rpl import RadarSignalProcessing"
            )

    # ------------------------------------------------------------------
    # Caricamento radar FFT
    # ------------------------------------------------------------------

    def _load_fft(self, sample_id: int) -> np.ndarray:
        """
        Carica la FFT pre-computata.

        REF: FFTRadNet/dataset/dataset.py
        File: radar_FFT/fft_{:06d}.npy
        Formato: numpy array complesso → concatena real+imag su axis=2.
        """
        fft_path = self.root / "radar_FFT" / f"fft_{sample_id:06d}.npy"
        raw = np.load(str(fft_path), allow_pickle=True)
        # raw è un array complesso (R_fft, D_fft, n_channels_per_antenna*n_ant)
        # Ritorna direttamente per la pipeline di processing
        return raw

    def _build_rae_tensor_from_fft(self, fft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Costruisce (tensor_rae2, rad_map) dalla FFT in modalità "fft_only".

        NOTA: questa è un'approssimazione senza DoA completo.
        In questa modalità:
          - La FFT ha shape (R_fft, D_fft, C_complex) dove C_complex = real+imag split
          - Calcoliamo la potenza |FFT|² → range-Doppler power
          - La RAD map è la proiezione (range × doppler), somma su antenne
          - Il tensore [R, A, E, 2] viene costruito con E=1 e A=n_range_cells
            (approssimazione BEV) poi interpolato alla griglia target

        Per una stima DoA completa usa mode="calibrated".
        """
        R_fft, D, C = fft.shape

        # Calcola potenza su tutti i canali
        # fft ha già real+imag su axis=2, quindi è reale o complesso
        if np.iscomplexobj(fft):
            power = np.abs(fft) ** 2    # (R_fft, D, C)
        else:
            # real+imag concatenati: prima metà real, seconda metà imag
            mid = C // 2
            power = fft[..., :mid] ** 2 + fft[..., mid:] ** 2   # (R_fft, D, mid)

        # RAD map: somma su antenne → (R_fft, D)
        rad_map_raw = power.mean(axis=-1)   # (R_fft, D)

        # Normalizzazione log-scale
        rad_map_raw = np.log1p(rad_map_raw)
        if rad_map_raw.max() > 0:
            rad_map_raw = rad_map_raw / rad_map_raw.max()

        # Interpola RAD map a (R=480, D) mantenendo D
        # Usiamo crop/pad per R:
        R_crop = min(R_fft, _R)
        rad_map = np.zeros((_R, rad_map_raw.shape[1]), dtype=np.float32)
        rad_map[:R_crop] = rad_map_raw[:R_crop]

        # Tensore [R, A, E, 2] approssimato (senza vera stima DoA):
        # Distribuiamo i range-Doppler su una griglia BEV piatta (E=1, A=R_fft)
        # QUESTA È UN'APPROSSIMAZIONE — da sostituire con la modalità "calibrated"
        d_star    = np.argmax(rad_map, axis=1)              # (R,) bin Doppler max
        intensity = rad_map[np.arange(_R), np.clip(d_star, 0, rad_map.shape[1]-1)]  # (R,)
        velocity  = (d_star.astype(np.float32) - rad_map.shape[1] // 2) * 0.04   # m/s (appross)
        velocity  = np.clip(velocity / 5.0, -1.0, 1.0)

        # Crea tensore [R, A, E, 2] con A=736, E=11
        # Distribuiamo i range su tutti gli azimuth (approssimazione BEV)
        tensor_rae2 = np.zeros((_R, _A, _E, 2), dtype=np.float32)
        # Centro azimuth = A//2 (direzione frontale); centro elevation = E//2
        az_c = _A // 2
        el_c = _E // 2
        tensor_rae2[:, az_c, el_c, 0] = intensity
        tensor_rae2[:, az_c, el_c, 1] = velocity

        return tensor_rae2, rad_map.astype(np.float32)

    # ------------------------------------------------------------------
    # Caricamento LiDAR e GT occupancy
    # ------------------------------------------------------------------

    def _load_lidar(self, sample_id: int) -> np.ndarray:
        """
        Carica il punto cloud LiDAR in formato Velodyne (N, 6).
        REF: FFTRadNet/dataset/dataset.py — laser_pcs usa DBReader
        Il ready-to-use ha laser_PCL/laser_PCL_{:06d}.npy.

        Formato: strutturato → unstructured (x, y, z, intensity, ring, ts).
        """
        from numpy.lib.recfunctions import structured_to_unstructured

        pc_path = self.root / "laser_PCL" / f"laser_PCL_{sample_id:06d}.npy"
        if not pc_path.exists():
            return np.zeros((0, 3), dtype=np.float32)

        raw = np.load(str(pc_path))
        if raw.dtype.names is not None:
            pc = structured_to_unstructured(raw).reshape(-1, len(raw.dtype.names))
        else:
            pc = raw.reshape(-1, -1) if raw.ndim == 1 else raw

        return pc[:, :3].astype(np.float64)   # solo xyz

    def _build_lidar_gt(self, lidar_xyz: np.ndarray) -> np.ndarray:
        """
        Costruisce l'occupancy GT [R, A, E] dal LiDAR.
        Applica prima Patchwork++ per rimozione suolo.
        """
        if len(lidar_xyz) == 0:
            return np.zeros((_R, _A, _E), dtype=np.float32)

        from src.alignment.ground_removal import GroundRemover
        remover = GroundRemover()
        clean, _ = remover.remove_ground_with_ego(lidar_xyz)

        if len(clean) == 0:
            return np.zeros((_R, _A, _E), dtype=np.float32)

        # Voxelizza con griglia RADIal
        occ = self.voxelizer.voxelize(clean[:, :3])
        return occ.astype(np.float32)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]

        # ── Tensore radar: usa sample corrente + 2 precedenti (id-1, id-2) ──
        t_rae2_frames = []
        rad_map_t = None

        for offset in [2, 1, 0]:   # t-2, t-1, t
            sid = max(self.sample_ids[0], sample_id - offset)
            fft = self._load_fft(sid)

            if self.proc == "calibrated" and self._rsp is not None:
                # TODO: implementa processing calibrato con RadarSignalProcessing
                # Richiede i file ADC binari, non disponibili nel ready-to-use.
                # Il ready-to-use ha già le FFT, che non supportano AoA completo.
                raise NotImplementedError(
                    "La modalità 'calibrated' richiede i file ADC binari, "
                    "non disponibili nel dataset 'ready-to-use'.\n"
                    "Scarica il dataset RADIal raw e usa rpl.RadarSignalProcessing.run(adc0,1,2,3)."
                )
            else:
                t_rae2, rad = self._build_rae_tensor_from_fft(fft)
                t_rae2_frames.append(t_rae2)
                if offset == 0:
                    rad_map_t = rad

        # Fusione temporale: (R, A, E, 6)
        t2, t1, t0 = t_rae2_frames[0], t_rae2_frames[1], t_rae2_frames[2]
        radar_cube = np.concatenate([t0, t1, t2], axis=-1)   # [t, t-1, t-2]

        # LiDAR GT
        lidar_xyz = self._load_lidar(sample_id)
        lidar_occ = self._build_lidar_gt(lidar_xyz)

        # Converti a tensori PyTorch
        sample = {
            "radar_cube": torch.from_numpy(
                radar_cube.transpose(3, 0, 1, 2)  # (6, R, A, E)
            ),
            "rad_map": torch.from_numpy(
                rad_map_t.T[np.newaxis, :, :]     # approssimazione: (1, D, R) o (D, R, A)
                # TODO: aggiusta shape a (D, R, A) una volta implementata modalità calibrata
            ),
            "lidar_occ": torch.from_numpy(lidar_occ),   # (R, A, E)
            "meta": {
                "sample_id": sample_id,
                "mode": self.mode,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
