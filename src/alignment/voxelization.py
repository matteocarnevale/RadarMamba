"""
Passo B5 — Voxelization → 3D Occupancy Grid (Ground Truth)
============================================================
REF: Paper Section 3.2, Step (5)
REF: RaDelft::non_uniform_voxelize_numpy + lidarpc_to_lidarcube

Il RaDelft usa griglie NON-UNIFORMI per gli assi (range, azimuth, elevation):
    range_axis    = np.arange(0.1004, 51.5, 0.1004)[10:-3]       # ~487 valori in metri
    azimuth_axis  = arcsin(linspace(-π, π, 256)[8:248] / (2π*0.4972))  # 240 valori in rad
    elevation_axis = arcsin(linspace(-π, π, 128)[47:81] / (2π*0.4972)) # 34 valori in rad

Usa np.searchsorted (come in data_preparation.py) per trovare il bin più vicino.
Dopo la voxelization applica flip (az + el) e transpose [E, R, A] → [R, A, E].

Per RADIal la griglia è più uniforme — usiamo la stessa funzione con assi uniformi.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Griglia non-uniforme per RaDelft (valori fisici, parametri reali)
# ------------------------------------------------------------------

def radelft_default_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restituisce gli assi NON-UNIFORMI del RaDelft come in get_default_params().
    Unità: range in metri, azimuth/elevation in RADIANTI.

    REF: RaDelft::get_default_params()

    NOTA SULLE DIMENSIONI:
        Il radarCube grezzo dal dataset ha forma [R=500, D=128, A=240].
        In get_default_params(), range_axis usa [10:-3] → 487 bin, ma questo
        è l'asse per la voxelizzazione del LiDAR (copre l'intervallo fisicamente
        valido). Il cubo radar grezzo ha 500 bin (incluse le celle marginali).
        Per far coincidere cubo radar e GT voxelizzato, usiamo 500 bin nel range_axis
        (i primi 500 valori dell'asse continuo, cioè da 0.1004m a ~50.2m).
    """
    # Range axis — 500 bin per matchare esattamente radarCube.shape[0]
    range_cell_size = 0.1004
    max_range       = 51.4242
    range_axis_full = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
    range_axis      = range_axis_full[:500]   # 500 bin = shape del radarCube

    # Azimuth axis (antenna spacing = 0.4972 × λ)
    angle_fft_size = 256
    wx_vec         = np.linspace(-np.pi, np.pi, angle_fft_size)[8:248]   # 240 bins
    azimuth_axis   = np.arcsin(np.clip(wx_vec / (2 * np.pi * 0.4972), -1.0, 1.0))

    # Elevation axis
    ele_fft_size   = 128
    wz_vec         = np.linspace(-np.pi, np.pi, ele_fft_size)[47:81]     # 34 bins
    elevation_axis = np.arcsin(np.clip(wz_vec / (2 * np.pi * 0.4972), -1.0, 1.0))

    return range_axis, azimuth_axis, elevation_axis


# ------------------------------------------------------------------
# Voxelizzazione non-uniforme (esatta dalla repo RaDelft)
# ------------------------------------------------------------------

def non_uniform_voxelize(
    point_cloud_sph: np.ndarray,
    range_axis:     np.ndarray,
    azimuth_axis:   np.ndarray,
    elevation_axis: np.ndarray,
) -> np.ndarray:
    """
    Voxelizza un punto cloud sferico [range, azimuth, elevation] su una griglia
    non-uniforme usando np.searchsorted.

    REF: data_preparation.non_uniform_voxelize_numpy (copia quasi esatta)

    Args:
        point_cloud_sph: (N, 3) — [range_m, az_rad, el_rad].
        range_axis:      (R,) — range bins in metri.
        azimuth_axis:    (A,) — azimuth bins in radianti.
        elevation_axis:  (E,) — elevation bins in radianti.

    Returns:
        voxel_grid: (R, A, E) float32 — 1 dove c'è almeno un punto, 0 altrimenti.
    """
    num_r = len(range_axis)
    num_a = len(azimuth_axis)
    num_e = len(elevation_axis)

    voxel_grid = np.zeros((num_r, num_a, num_e), dtype=np.float32)

    if len(point_cloud_sph) == 0:
        return voxel_grid

    # Trova bin più vicino per ogni asse con searchsorted
    r_idx = np.searchsorted(range_axis,     point_cloud_sph[:, 0], side="left")
    a_idx = np.searchsorted(azimuth_axis,   point_cloud_sph[:, 1], side="left")
    e_idx = np.searchsorted(elevation_axis, point_cloud_sph[:, 2], side="left")

    # Filtra punti fuori dalla griglia
    valid = (
        (r_idx >= 0) & (r_idx < num_r) &
        (a_idx >= 0) & (a_idx < num_a) &
        (e_idx >= 0) & (e_idx < num_e)
    )
    r_idx, a_idx, e_idx = r_idx[valid], a_idx[valid], e_idx[valid]
    pc_valid = point_cloud_sph[valid]

    # Corregge verso il bin più vicino (non sempre il bin sinistro)
    # REF: data_preparation.py lines 91-103
    def _correct_idx(idx, axis, values):
        condition = (idx > 0) & (
            (idx == len(axis)) |
            (np.abs(values - axis[np.clip(idx - 1, 0, len(axis)-1)]) <
             np.abs(values - axis[np.clip(idx, 0, len(axis)-1)]))
        )
        idx = idx.copy()
        idx[condition] -= 1
        return idx

    r_idx = _correct_idx(r_idx, range_axis,     pc_valid[:, 0])
    a_idx = _correct_idx(a_idx, azimuth_axis,   pc_valid[:, 1])
    e_idx = _correct_idx(e_idx, elevation_axis, pc_valid[:, 2])

    # Clip finale per sicurezza
    r_idx = np.clip(r_idx, 0, num_r - 1)
    a_idx = np.clip(a_idx, 0, num_a - 1)
    e_idx = np.clip(e_idx, 0, num_e - 1)

    voxel_grid[r_idx, a_idx, e_idx] = 1.0
    return voxel_grid


def lidarpc_to_lidarcube(
    lidar_xyz: np.ndarray,
    range_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    elevation_axis: np.ndarray,
) -> np.ndarray:
    """
    Converte un punto cloud LiDAR cartesiano nella griglia occupancy [R, A, E].

    Replicates RaDelft::lidarpc_to_lidarcube (non-BEV path, output [E, R, A] → [R, A, E]).

    REF: data_preparation.lidarpc_to_lidarcube

    Args:
        lidar_xyz:     (N, 3) — [x, y, z] cartesiani nel frame del sensore.
        range/azimuth/elevation_axis: assi non-uniformi.

    Returns:
        cube: (R, A, E) float32 — griglia di occupancy.
    """
    from src.alignment.coordinate_transform import cartesian_to_spherical_rad

    if len(lidar_xyz) == 0:
        return np.zeros((len(range_axis), len(azimuth_axis), len(elevation_axis)), dtype=np.float32)

    # Cartesiano → sferico (range_m, az_rad, el_rad)
    sph = cartesian_to_spherical_rad(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2])

    # Voxelizza su griglia non-uniforme (output: [R, A, E])
    cube = non_uniform_voxelize(sph, range_axis, azimuth_axis, elevation_axis)

    # Flip degli assi azimuth e elevation (come in data_preparation.py)
    # REF: data_preparation.py lines 329-330
    cube = np.flip(cube, axis=1)  # flip azimuth
    cube = np.flip(cube, axis=2)  # flip elevation

    return cube.copy().astype(np.float32)


# ------------------------------------------------------------------
# Classe wrapper orientata agli oggetti
# ------------------------------------------------------------------

class Voxelizer:
    """
    Voxelizzatore configurabile per RADIal e RaDelft.
    Usa assi non-uniformi come in RaDelft::lidarpc_to_lidarcube.
    """

    def __init__(
        self,
        range_axis:     np.ndarray,
        azimuth_axis:   np.ndarray,
        elevation_axis: np.ndarray,
    ) -> None:
        self.range_axis     = range_axis
        self.azimuth_axis   = azimuth_axis
        self.elevation_axis = elevation_axis
        self.R = len(range_axis)
        self.A = len(azimuth_axis)
        self.E = len(elevation_axis)

    @classmethod
    def for_radelft(cls) -> "Voxelizer":
        """Crea un Voxelizer con i parametri reali di RaDelft."""
        r, a, e = radelft_default_axes()
        return cls(r, a, e)

    @classmethod
    def for_radial(cls) -> "Voxelizer":
        """
        Crea un Voxelizer per RADIal.
        Grid: [R=480, A=736, E=11] — range [0,50]m, az [-75°,75°], el [-4°,6°].
        Usiamo assi uniformi dato che RADIal non fornisce un CalibrationTable per la griglia.
        """
        R, A, E = 480, 736, 11
        range_axis     = np.linspace(0.0, 50.0, R + 1)[1:]              # (R,) in m
        azimuth_axis   = np.linspace(np.radians(-75), np.radians(75), A)  # (A,) in rad
        elevation_axis = np.linspace(np.radians(-4),  np.radians(6),   E)  # (E,) in rad
        return cls(range_axis, azimuth_axis, elevation_axis)

    def voxelize(self, lidar_xyz: np.ndarray) -> np.ndarray:
        """
        Pipeline completa: cartesiano [x,y,z] → occupancy [R, A, E].

        Args:
            lidar_xyz: (N, 3) — coordinate cartesiane.

        Returns:
            cube: (R, A, E) float32.
        """
        return lidarpc_to_lidarcube(
            lidar_xyz,
            self.range_axis,
            self.azimuth_axis,
            self.elevation_axis,
        )

    @property
    def occupancy_shape(self) -> tuple[int, int, int]:
        return (self.R, self.A, self.E)

    def point_density(self, occupancy: np.ndarray) -> float:
        return float(occupancy.sum()) / occupancy.size
