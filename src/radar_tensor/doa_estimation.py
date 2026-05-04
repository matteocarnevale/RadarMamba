"""
Direction of Arrival (DoA) Estimation
========================================
Stima azimuth ed elevation dei bersagli radar usando beamforming 3D
sul cubo virtuale antennas × range × Doppler.

REF: Paper Section 3.2:
     "CFAR-based noise filtering and Direction-of-Arrival (DoA) estimation
      produce 4D radar point clouds comprising range, azimuth, elevation,
      velocity, and intensity (R, A, E, D, I) information."

     Blue path (nostra implementazione):
     "Our improved radar processing methodology removes CFAR and directly
      builds a 4D radar tensor [R, A, E, D], representing the resolution of
      range, azimuth, elevation, and Doppler in the radar coordinate system.
      For computational efficiency, only the strongest reflection points and
      their Doppler velocities are retained, yielding a compact tensor [R, A, E, 2]."

Algoritmo:
    Il metodo più semplice ed efficiente per radar MIMO è il 3D-FFT beamforming:
    - Azimuth FFT sulle antenne virtuali orizzontali
    - Elevation FFT sulle antenne virtuali verticali
    Output: cubo [R, A, E, D] di amplitudine (complessa) o potenza.

Note su RADIal:
    192 antenne virtuali (16 Rx × 12 Tx).
    La geometria dell'array determina la risoluzione angolare.
    La libreria SignalProcessing del repo RADIal fa già questo.

Note su RaDelft:
    MMWCAS-RF-EVM: 4 chip × 3 TX × 4 RX = 48 virtual Rx (+ cascading).
    Consulta la documentazione hardware per il layout dell'array.
"""

from __future__ import annotations

import numpy as np


class DoAEstimator:
    """
    Stima di DoA tramite 3D FFT beamforming su array MIMO.

    Produce un cubo 4D di potenza [R, A, E, D] dal quale costruiamo
    il tensore radar [R, A, E, 2] prendendo i picchi.
    """

    def __init__(
        self,
        radar_cfg: dict,
        grid_cfg: dict,
        antenna_positions: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            radar_cfg:  parametri radar dal YAML (n_rx, n_tx, n_virtual, ecc.).
            grid_cfg:   griglia [R, A, E, D] dal YAML.
            antenna_positions: np.ndarray, shape (n_virtual, 3) — posizioni delle
                               antenne virtuali in unità di mezza lunghezza d'onda.
                               Se None, assume array ULA (Uniform Linear Array) per
                               entrambe le dimensioni (approssimazione).
        """
        self.cfg = radar_cfg
        self.grid = grid_cfg
        self.antenna_positions = antenna_positions

    # ------------------------------------------------------------------
    # Beamforming via 3D-FFT (metodo semplice)
    # ------------------------------------------------------------------

    def beamform_3d_fft(
        self,
        virtual_array: np.ndarray,
    ) -> np.ndarray:
        """
        Beamforming 3D-FFT sull'array virtuale MIMO.

        Args:
            virtual_array: np.ndarray, shape (n_doppler, n_virtual, n_range)
                           — segnale complesso dopo range+Doppler FFT,
                             riorganizzato sulle antenne virtuali.

        Returns:
            radar_cube: np.ndarray, shape (n_range, n_azimuth, n_elevation, n_doppler)
                        — cubo di potenza |.|².

        TODO:
            1. Riorganizza le antenne virtuali in una griglia 2D (azimuth × elevation):
                   azimuth_array   = virtual_array[:, az_indices, :]
                   elevation_array = virtual_array[:, el_indices, :]
               La riorganizzazione dipende dal layout fisico delle antenne — consulta
               i datasheet del radar e le note del dataset.

            2. Applica window function sulle dimensioni angolari (Hann o Chebyshev).

            3. Azimuth FFT (axis=1) → dimensione A = grid_cfg["A"].
            4. Elevation FFT (axis=2) → dimensione E = grid_cfg["E"].
            5. Calcola potenza: |cubo|² → forma (D, A, E, R).
            6. Trasponi in (R, A, E, D).
            7. Normalizza (opzionale).

            Alternativa più accurata: usa MUSIC o Capon beamformer
            per una migliore risoluzione angolare (ma più lento).
        """
        raise NotImplementedError(
            "TODO: implementa 3D-FFT beamforming.\n"
            "Passi: azimuth FFT + elevation FFT sulle antenne virtuali."
        )

    # ------------------------------------------------------------------
    # Costruzione tensore [R, A, E, 2] dal cubo 4D
    # ------------------------------------------------------------------

    def cube_to_tensor(
        self,
        radar_cube: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estrae il tensore [R, A, E, 2] dal cubo [R, A, E, D].

        REF: Paper Section 3.2:
             "Only the strongest reflection points and their Doppler velocities
              are retained, yielding a compact tensor [R, A, E, 2]."

        Per ogni voxel (r, a, e):
            - canale 0: intensità = max_{d} |cube[r,a,e,d]|  (riflessione più forte)
            - canale 1: velocità  = argmax_{d} |cube[r,a,e,d]| convertito in m/s

        Args:
            radar_cube: np.ndarray, shape (R, A, E, D) — cubo di potenza.

        Returns:
            tensor_rae2: np.ndarray, shape (R, A, E, 2) — tensore [intensità, Doppler].
            rad_map:     np.ndarray, shape (R, A, D)    — proiezione RA (marginalizzata su E).

        TODO:
            1. intensity = radar_cube.max(axis=-1)  → (R, A, E)

            2. doppler_idx = radar_cube.argmax(axis=-1)  → (R, A, E)

            3. Converti doppler_idx in velocità reale (m/s):
                   velocita = (doppler_idx - D//2) * velocity_resolution
                   dove velocity_resolution = lambda / (2 * n_chirps * T_chirp)
                   (lambda = wavelength, T_chirp = chirp period)

            4. tensor_rae2 = np.stack([intensity, velocita], axis=-1)  → (R, A, E, 2)

            5. rad_map = radar_cube.max(axis=2)  → (R, A, D)
               Marginalizziamo su E prendendo il massimo — così la mappa RAD
               rappresenta i bersagli più forti su ogni cella (R, A, D).

            6. Normalizza intensity e velocity se necessario (es. log-scale per intensity).
        """
        raise NotImplementedError(
            "TODO: estrai tensore [R,A,E,2] e mappa RAD dal cubo 4D."
        )

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def estimate(
        self,
        virtual_array: np.ndarray,
    ) -> dict:
        """
        Stima DoA completa: dall'array virtuale al tensore radar + RAD map.

        Args:
            virtual_array: np.ndarray, shape (n_doppler, n_virtual, n_range).

        Returns:
            dict con chiavi:
                "tensor_rae2": np.ndarray (R, A, E, 2)
                "rad_map":     np.ndarray (R, A, D)
                "radar_cube":  np.ndarray (R, A, E, D) — opzionale per debug
        """
        radar_cube = self.beamform_3d_fft(virtual_array)
        tensor_rae2, rad_map = self.cube_to_tensor(radar_cube)

        return {
            "tensor_rae2": tensor_rae2,
            "rad_map": rad_map,
            "radar_cube": radar_cube,
        }
