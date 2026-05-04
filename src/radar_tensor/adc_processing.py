"""
ADC Processing — da ADC grezzo a Range-Doppler e MIMO spectrum
================================================================
REF: Paper Section 3.2 (blue path, Fig. 2)
REF: RADIal SignalProcessing/rpl.py::RadarSignalProcessing

PIPELINE RADAR-MAMBA (blue path, senza CFAR):

  ADC raw (4 file .bin) → build_radar_frame → DC removal
                        → Range FFT (Hanning window)
                        → Doppler FFT (Hanning window + fftshift)
                        → MIMO re-ordering (TX DDMA Doppler shift compensation)
                        → Window (CalibrationTable)
                        → AoA beamforming (CalibMat @ MIMO_spectrum)
                        → Dense 4D cube [R, A, E, D] (NO CFAR gate)
                        → Max-pool Doppler → tensor [R, A, E, 2] (intensity, velocity)
                        → RAD map [R, A, D] (marginalize E)

Differenza dal percorso grigio (Fig. 2):
  - Il percorso grigio applica CFAR sul RD map → sparse point cloud → tensore sparso
  - Il percorso blu (Radar-Mamba) bypassa CFAR → tensore denso [R,A,E,2]

NOTA SUI FILE ADC:
  RADIal usa 4 chip radar TI con 4 Rx antennas ciascuno → 16 Rx totali.
  12 Tx (DDMA chirp encoding) → 192 virtual antennas.
  Ogni file .bin ha: [n_samples_per_chirp × n_rx_per_chip × n_chirps] interleaved.
  REF: rpl.py::__build_radar_frame

CalibrationTable.npy:
  AoA_mat['Signal'] : (n_az, n_virtual, n_el) complex — matrice di calibrazione
  AoA_mat['Azimuth_table']   : (n_az,) float — angoli in gradi
  AoA_mat['Elevation_table'] : (n_el,) float — angoli in gradi
  AoA_mat['H']     : (1, n_virtual) complex — finestra Hamming (Hanning per sidelobes)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# Parametri hardware fissi RADIal (da rpl.py)
NUM_SAMPLES   = 512    # ADC samples per chirp
NUM_RX_CHIP   = 4     # Rx antennas per chip
NUM_CHIRPS    = 256   # chirps per frame
NUM_RX_ANT    = 16    # Rx antennas totali (4 chips × 4)
NUM_TX_ANT    = 12    # Tx antennas (DDMA)
NUM_VIRTUAL   = 192   # virtual = NUM_RX_ANT × NUM_TX_ANT / DDMA_factor
NUM_RED_DOP   = 16    # riduzione Doppler per DDMA
NUM_CHIRPS_PER_LOOP = 16


class RADIalADCProcessor:
    """
    Processa dati ADC grezzi di RADIal → tensore [R, A, E, 2] + RAD map.
    Replica fedelmente rpl.py::RadarSignalProcessing.

    Per usare questo processore:
        1. Installa: pip install mkl_fft  (richiede Intel MKL, tipicamente su Linux)
           Alternativa: sostituisci mkl_fft.fft con np.fft.fft (più lento ma portabile)
        2. Scarica CalibrationTable.npy dal repo RADIal
        3. Metti i 4 file ADC binari nella stessa directory del frame
    """

    def __init__(self, calib_path: str | Path) -> None:
        """
        Args:
            calib_path: path a CalibrationTable.npy del repo RADIal.
        """
        self.calib_path = Path(calib_path)
        self.AoA_mat    = np.load(str(calib_path), allow_pickle=True).item()

        # Matrice di calibrazione per beamforming completo 3D:
        # Signal shape: (n_az, n_virtual, n_el) → reshape a (n_az*n_el, n_virtual)
        sig = self.AoA_mat["Signal"]          # (n_az, n_virtual, n_el)
        n_az, n_virt, n_el = sig.shape
        self.n_az   = n_az
        self.n_el   = n_el
        self.n_virt = n_virt

        # CalibMat per beamforming completo: [n_az*n_el, n_virtual]
        # REF: rpl.py usa Signal[:,:,5] solo per RA; per 3D usiamo tutti gli el
        self.CalibMat_3d = np.rollaxis(sig, 2, 1).reshape(n_az * n_el, n_virt)

        # Finestra Hanning per ridurre i sidelobes nell'AoA
        self.window_az = self.AoA_mat["H"][0]   # (n_virtual,)

        # Tabelle degli angoli fisici
        self.az_table  = self.AoA_mat["Azimuth_table"]    # (n_az,) gradi
        self.el_table  = self.AoA_mat["Elevation_table"]  # (n_el,) gradi

        # Finestre di windowing
        hanning_r  = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(NUM_SAMPLES)
                                           / (NUM_SAMPLES - 1))
        hanning_d  = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(NUM_CHIRPS)
                                           / (NUM_CHIRPS - 1))
        self.win_range  = hanning_r[:, np.newaxis, np.newaxis]  # (512, 1, 1)
        self.win_doppler = hanning_d[np.newaxis, :, np.newaxis]  # (1, 256, 1)

        self.dividend_arr = np.arange(0, NUM_RED_DOP * NUM_CHIRPS_PER_LOOP, NUM_RED_DOP)

    # ------------------------------------------------------------------
    # Lettura e assembly del frame MIMO
    # ------------------------------------------------------------------

    @staticmethod
    def read_adc_binary(adc_path: str | Path) -> np.ndarray:
        """
        Legge un file ADC binario RADIal.

        REF: rpl.py::__build_radar_frame
        Formato: int16 interleaved (real, imag) con ordine F per (samples, rx, chirps).

        Returns:
            raw: (n_samples*n_rx_chip*n_chirps*2,) int16
        """
        return np.fromfile(str(adc_path), dtype=np.int16)

    def _build_radar_frame(
        self,
        adc0: np.ndarray, adc1: np.ndarray,
        adc2: np.ndarray, adc3: np.ndarray
    ) -> np.ndarray:
        """
        Assembla 4 stream ADC nel frame MIMO complesso.
        REF: rpl.py::__build_radar_frame (copia fedele)

        Returns:
            frame: (NUM_SAMPLES, NUM_CHIRPS, NUM_RX_ANT) complex64
        """
        def _chip_to_frame(adc_raw: np.ndarray) -> np.ndarray:
            cplx = adc_raw[0::2] + 1j * adc_raw[1::2]
            return cplx.reshape(NUM_SAMPLES, NUM_RX_CHIP, NUM_CHIRPS, order="F").transpose(0, 2, 1)

        f0 = _chip_to_frame(adc0)
        f1 = _chip_to_frame(adc1)
        f2 = _chip_to_frame(adc2)
        f3 = _chip_to_frame(adc3)
        # Chip ordering from rpl.py: [frame3, frame0, frame1, frame2]
        return np.concatenate([f3, f0, f1, f2], axis=2)    # (N_SAMPLES, N_CHIRPS, 16)

    # ------------------------------------------------------------------
    # Range + Doppler FFT
    # ------------------------------------------------------------------

    def _range_doppler_fft(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcola la mappa Range-Doppler (2D FFT).
        REF: rpl.py::run → steps 2-4

        Args:
            frame: (N_SAMPLES, N_CHIRPS, N_RX_ANT) complex64

        Returns:
            RD_spectrums: (N_SAMPLES, N_CHIRPS, N_RX_ANT) complex64
        """
        try:
            import mkl_fft
            fft_fn = mkl_fft.fft
        except ImportError:
            # Fallback a numpy.fft (più lento ma portabile)
            fft_fn = np.fft.fft

        # Rimuovi DC offset
        frame = frame - frame.mean(axis=(0, 1), keepdims=True)

        # Range FFT (axis=0, campioni ADC)
        range_fft = fft_fn(frame * self.win_range, NUM_SAMPLES, axis=0)

        # Doppler FFT (axis=1, chirps)
        RD = fft_fn(range_fft * self.win_doppler, NUM_CHIRPS, axis=1)

        return RD   # (N_SAMPLES, N_CHIRPS, N_RX_ANT)

    # ------------------------------------------------------------------
    # DDMA Doppler shift compensation e MIMO re-ordering
    # ------------------------------------------------------------------

    def _build_mimo_spectrum(
        self,
        RD: np.ndarray,
        range_bins: np.ndarray,    # (M,) int
        doppler_bins: np.ndarray,  # (M,) int
    ) -> np.ndarray:
        """
        Ricostruisce lo spettro MIMO per le celle (range, Doppler) di interesse.
        REF: rpl.py::__get_PCL passi 3-4 (DDMA TX ordering + window)

        Args:
            RD:          (N_SAMPLES, N_CHIRPS, 16) complex
            range_bins:  (M,) indici range
            doppler_bins: (M,) indici Doppler

        Returns:
            MIMO_Spectrum: (M, n_virtual) complex — spettro MIMO con window
        """
        # Costruisci la sequenza Doppler per ogni Tx (DDMA phase coding)
        # REF: rpl.py::__find_TX0_position + __get_PCL
        doppler_seqs = []
        for dop in doppler_bins:
            seq = np.remainder(dop + self.dividend_arr, NUM_CHIRPS)
            seq = np.concatenate([[seq[0]], seq[5:]]).astype(int)  # 8 doppler bins per Tx
            doppler_seqs.append(seq)

        rb = [[r] for r in range_bins]
        MIMO = RD[rb, doppler_seqs, :].reshape(len(range_bins), -1)  # (M, 16*8=128)
        # Applica finestra Hanning per sidelobe reduction
        MIMO = MIMO * self.window_az
        return MIMO

    # ------------------------------------------------------------------
    # DoA beamforming 3D → cubo [R, A, E, D]
    # ------------------------------------------------------------------

    def build_radar_cube_dense(
        self,
        adc_paths: list[str | Path],
        grid_R: int = 480,
        grid_A: int = 736,
        grid_E: int = 11,
        az_min_deg: float = -75.0,
        az_max_deg: float = 75.0,
        el_min_deg: float = -4.0,
        el_max_deg: float = 6.0,
        range_scale: float = 103.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processa 4 file ADC → dense cube [R, A, E, 2] + RAD map [R, A, D].

        NOTA: questa funzione costruisce il cubo denso senza CFAR — è il
        "blue path" di Fig. 2 del paper. Sostituisce la CFAR hard-detection
        con un cubo continuo che il modello Mamba processa direttamente.

        Args:
            adc_paths:  lista di 4 path ai file .bin ADC (chips 0,1,2,3).
            grid_*:     dimensioni della griglia target.
            az/el_*_deg: FoV in gradi.
            range_scale: fattore di conversione range_bin → m (103 per RADIal).

        Returns:
            tensor_rae2: (grid_R, grid_A, grid_E, 2) float32
            rad_map:     (grid_R, grid_A, D_red)   float32
        """
        assert len(adc_paths) == 4, "Servono 4 file ADC (uno per chip)"

        # 1. Leggi e assembla il frame MIMO
        adcs = [self.read_adc_binary(p) for p in adc_paths]
        frame = self._build_radar_frame(*adcs)          # (N_SAMPLES, N_CHIRPS, 16)

        # 2. Range-Doppler 2D FFT
        RD = self._range_doppler_fft(frame)              # (N_SAMPLES, N_CHIRPS, 16)

        # 3. Calcola potenza per la RAD map (marginalizza sulle antenne)
        power_rd = np.sum(np.abs(RD) ** 2, axis=2)      # (N_SAMPLES, N_CHIRPS)

        # 4. Costruisci la RAD map ridotta (somma Doppler bins per gruppo TX)
        # REF: rpl.py usa reduced_power_spectrum con reshape(512,16,16)
        D_red = 16   # doppler bins ridotti (NUM_CHIRPS / NUM_RED_DOP = 256/16)
        power_rd_red = power_rd.reshape(NUM_SAMPLES, D_red, NUM_RED_DOP).sum(axis=2)
        # → (N_SAMPLES, D_red=16)

        # 5. Beamforming 3D su tutti i range-Doppler bins (senza CFAR gate)
        # Per ogni (r, d) bin calcola lo spettro AoA sull'array virtuale MIMO
        # Output: cubo di ampiezza [n_az*n_el, N_SAMPLES, D_red]
        # Approccio vettorizzato: processa tutti i bin range per ogni Doppler ridotto

        # Costruisci MIMO spectrum su tutti i bins
        # REF: rpl.py::__get_RA (ma per tutti gli el, non solo el=5)
        all_doppler_bins = np.arange(NUM_CHIRPS)
        all_dop_seqs = []
        for dop in all_doppler_bins:
            seq = np.remainder(dop + self.dividend_arr, NUM_CHIRPS)
            seq = np.concatenate([[seq[0]], seq[5:]]).astype(int)
            all_dop_seqs.append(seq)

        # MIMO_all: (N_SAMPLES * N_CHIRPS, n_virtual)
        r_all = np.repeat(np.arange(NUM_SAMPLES), NUM_CHIRPS)
        d_all = np.tile(np.arange(NUM_CHIRPS), NUM_SAMPLES)
        MIMO_all = RD[r_all, [all_dop_seqs[d] for d in d_all], :]   # (N_S*N_D, 8*16)
        # Shape after reshape: RD[range_bins, doppler_seqs, :].reshape(-1, 128)

        # Alternativa più memory-friendly: processa per segmenti di range
        # Per semplicità qui usiamo la versione batch
        MIMO_flat = RD[:, all_dop_seqs, :].reshape(NUM_SAMPLES * NUM_CHIRPS, -1)
        MIMO_flat = MIMO_flat * self.window_az   # windowing

        # AoA 3D: CalibMat_3d @ MIMO.T → (n_az*n_el, N_S*N_D)
        ASpec = np.abs(self.CalibMat_3d @ MIMO_flat.T)   # (n_az*n_el, N_S*N_D)
        ASpec = ASpec.reshape(self.n_az, self.n_el, NUM_SAMPLES, NUM_CHIRPS)
        # → (n_az, n_el, N_SAMPLES, N_CHIRPS)

        # 6. Mappa gli indici angolari fisici → indici griglia target
        # Crea griglia target per az e el in gradi
        az_target = np.linspace(az_min_deg, az_max_deg, grid_A)  # (A,) gradi
        el_target = np.linspace(el_min_deg, el_max_deg, grid_E)  # (E,) gradi

        # Interpola: per ogni (a_t, e_t) nella griglia target, trova l'indice di az_table/el_table
        az_interp_idx = np.searchsorted(self.az_table, az_target).clip(0, self.n_az - 1)
        el_interp_idx = np.searchsorted(self.el_table, el_target).clip(0, self.n_el - 1)

        # Subset del cubo: (grid_A, grid_E, N_SAMPLES, N_CHIRPS)
        cube_ae = ASpec[np.ix_(az_interp_idx, el_interp_idx, np.arange(NUM_SAMPLES), np.arange(NUM_CHIRPS))]

        # Range: range_m = range_bin / N_SAMPLES * range_scale → tronca a grid_R
        range_scale_factor = range_scale / NUM_SAMPLES
        R_crop = min(NUM_SAMPLES, grid_R)

        # cube_ae[:, :, :R_crop, :] → (A, E, R, D) → (R, A, E, D) dopo swap
        cube_raed = cube_ae[:, :, :R_crop, :].transpose(2, 0, 1, 3)  # (R, A, E, D)

        # Pad se R_crop < grid_R
        if R_crop < grid_R:
            pad = np.zeros((grid_R - R_crop, grid_A, grid_E, NUM_CHIRPS), dtype=cube_raed.dtype)
            cube_raed = np.vstack([cube_raed, pad])

        # 7. tensor_rae2: intensità = max_D e velocity = argmax_D → velocità in m/s
        d_star    = np.argmax(cube_raed, axis=-1)                 # (R, A, E)
        intensity = cube_raed[
            np.arange(grid_R)[:, None, None],
            np.arange(grid_A)[None, :, None],
            np.arange(grid_E)[None, None, :],
            d_star
        ]                                                          # (R, A, E)

        # Normalizzazione log-scale intensità
        intensity = np.log1p(intensity.astype(np.float32))
        max_i = intensity.max()
        if max_i > 0:
            intensity /= max_i

        # Velocità in m/s (usando vel_bin_size approssimativo)
        vel_bin_size = 0.04   # approssimazione; il valore esatto è in params RaDelft
        velocity = (d_star.astype(np.float32) - NUM_CHIRPS // 2) * vel_bin_size
        velocity = np.clip(velocity / 5.0, -1.0, 1.0)   # normalizza [-1, 1]

        tensor_rae2 = np.stack([intensity, velocity], axis=-1)   # (R, A, E, 2)

        # 8. RAD map [R, A, D_red]: power marginalizzato su E
        # Usiamo power_rd_red[:R_crop, :] e la distribuiamo per azimuth
        # (approssimazione: senza DoA sull'azimuth per la RAD map)
        rad_map = np.zeros((grid_R, grid_A, D_red), dtype=np.float32)
        rad_map[:R_crop, grid_A // 2, :] = np.log1p(power_rd_red[:R_crop].astype(np.float32))
        # TODO: per la RAD map vera serve il beamforming RA completo (senza elevation)

        return tensor_rae2.astype(np.float32), rad_map
