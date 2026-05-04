"""
RaDelft Dataset — PyTorch Dataset (versione temporale, 3 frame)
================================================================
REF: https://github.com/RaDelft/RaDelft-Dataset
     machine_learning_python/loaders/rad_cube_loader.py::RADCUBE_DATASET_TIME

════════════════════════════════════════════════════════════════
COSA TI SERVE DAL DATASET  (risposta alla domanda)
════════════════════════════════════════════════════════════════
NON hai bisogno dell'ADC raw!

Il dataset RaDelft fornisce dati GIÀ PROCESSATI che corrispondono
esattamente all'output del "blue path" di Fig. 2 del paper:

  Pow_Frame_{idx}.mat["radarCube"]      → [R=500, D=128, A=240]
  │  ← già applicate: Range FFT + Doppler FFT + AoA azimuth beamforming
  │  ← D=128 = Doppler velocity bins
  │  ← A=240 = azimuth bins dopo DoA beamforming

  Ele_Frame_{idx}.mat["elevationIndex"] → [R=500, A=240]
  │  ← per ogni cella (r, a): indice del bin di elevation dominante (0-33)
  │  ← E=34 bin fisici di elevation, prodotto dalla DoA in elevation

  rosDS/rslidar_points_clean/*.npy      → (N, 3) [x, y, z]
     ← LiDAR GIÀ GROUND-REMOVED con Patchwork++ (non applicarlo di nuovo!)
     ← Formato rs_lidar_clean: reshape(-1, 3) direttamente

Da questi 3 file costruiamo:
  tensor [R=500, A=240, E=34, 2]  (intensity, normalized_velocity) per frame
  tensor [R=500, A=240, E=34, 6]  dopo fusione temporale 3 frame (input modello)
  RAD map [R=500, A=240, D=128]   (input Doppler backbone)
  GT occ  [R=500, A=240, E=34]    dal LiDAR voxelizzato

════════════════════════════════════════════════════════════════
Struttura cartelle del dataset:
    data/raw/radelft/
    └── Scene{N}/
        ├── RadarCubes/
        │   ├── Pow_Frame_{idx}.mat   → radarCube: [500, 128, 240] float32
        │   ├── Ele_Frame_{idx}.mat   → elevationIndex: [500, 240] float32
        │   ├── DopFold_Frame_{idx}.mat (opzionale)
        │   └── timestamps.mat        → unixDateTime[frame_num-1][0] in secondi
        └── rosDS/
            ├── rslidar_points_clean/ → *.npy (N,3) float32 [x,y,z] GIÀ PULITO
            └── ueye_left_image_rect_color/ (non usata)

Split (paper Sec 4.1 + repo RaDelft):
    Train: Scene 1, 3, 4, 5, 7   (90% per scene)
    Val:   Scene 1, 3, 4, 5, 7   (10% per scene, ogni 10° frame)
    Test:  Scene 2, 6             (tutti i frame)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from src.alignment.voxelization import Voxelizer, radelft_default_axes


# ── Parametri hardware RaDelft (da data_preparation.py::get_default_params) ──
_VEL_FFT_SIZE  = 128             # bin Doppler (D)
_VEL_BIN_SIZE  = 0.04607058455831936   # m/s per bin Doppler
_POWER_NORM    = 8998.5576       # normalizzazione potenza (max empirico)
_ELEV_N_BINS   = 34              # bin fisici di elevation (E)
_AZIMUTH_OFFSET_DEG = 7.0        # rotazione LiDAR→Radar attorno all'asse z
_X_OFFSET_M    = 0.0             # traslazione x in metri
_Y_OFFSET_M    = 0.0             # traslazione y in metri


def _get_ts_paths(directory: str) -> dict[int, str]:
    """
    Replica data_preparation.get_timestamps_and_paths.
    Ritorna {timestamp_ns: path} per ogni .npy nella directory.
    """
    out = {}
    for fname in os.listdir(directory):
        if fname.endswith(".npy"):
            parts = fname.split(".")
            if len(parts) >= 2:
                try:
                    ts = int(parts[0]) * 10**9 + int(parts[1])
                    out[ts] = os.path.join(directory, fname)
                except ValueError:
                    pass
    return out


def _closest_ts(new_ts: int, ts_dict: dict[int, str]) -> int:
    return min(ts_dict.keys(), key=lambda t: abs(t - new_ts))


def _rot_z(deg: float) -> np.ndarray:
    """Matrice di rotazione 3D attorno all'asse z."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]], dtype=np.float64)


class RaDelftDataset(Dataset):
    """
    Dataset PyTorch per RaDelft — versione temporale (3 frame consecutivi).
    Replica RADCUBE_DATASET_TIME adattato per Radar-Mamba.

    NON ha bisogno di ADC raw: usa Pow_Frame + Ele_Frame + LiDAR clean.
    """

    TRAIN_SCENES = [1, 3, 4, 5, 7]
    TEST_SCENES  = [2, 6]

    def __init__(
        self,
        dataset_path:       str | Path,
        mode:               str   = "train",   # "train" | "val" | "test"
        n_frames:           int   = 3,
        normalize_velocity: bool  = True,
        vel_max_mps:        float = 5.0,       # clipping per normalizzazione Doppler
        transform=None,
    ) -> None:
        assert mode in ("train", "val", "test")

        self.root     = Path(dataset_path)
        self.mode     = mode
        self.n_frames = n_frames
        self.norm_vel = normalize_velocity
        self.vel_max  = vel_max_mps
        self.transform = transform

        # Assi fisici non-uniformi RaDelft
        self.range_axis, self.az_axis, self.el_axis = radelft_default_axes()
        self.R = len(self.range_axis)   # 500
        self.A = len(self.az_axis)      # 240
        self.E = len(self.el_axis)      # 34
        self.D = _VEL_FFT_SIZE          # 128

        # Voxelizzatore: LiDAR cartesiano → occupancy [R, A, E]
        self.voxelizer = Voxelizer(self.range_axis, self.az_axis, self.el_axis)

        # Matrice di rotazione calibrazione LiDAR→Radar
        self._R_mat = _rot_z(_AZIMUTH_OFFSET_DEG)

        # Costruisci indice dei campioni
        scenes = self.TRAIN_SCENES if mode in ("train", "val") else self.TEST_SCENES
        self.samples = self._build_index(scenes)

    # ------------------------------------------------------------------
    # Indice dei campioni (gruppi di 3 frame consecutivi)
    # ------------------------------------------------------------------

    def _build_index(self, scenes: list[int]) -> list[list[dict]]:
        """
        Replica RADCUBE_DATASET_TIME.__init__:
        Raggruppa i frame in triplet [t-2, t-1, t] consecutive.
        """
        aux = []   # lista piatta di frame singoli

        for scene_num in scenes:
            scene_dir = self.root / f"Scene{scene_num}"
            cubes_dir = scene_dir / "RadarCubes"
            lidar_dir = scene_dir / "rosDS" / "rslidar_points_clean"

            if not cubes_dir.exists():
                import warnings
                warnings.warn(f"Scene{scene_num} non trovata in {cubes_dir} — saltata")
                continue

            # Frame disponibili
            pow_files  = [f for f in os.listdir(str(cubes_dir)) if "Pow_Frame" in f]
            frame_nums = sorted([int(f.split("_")[-1].split(".")[0]) for f in pow_files])
            if not frame_nums:
                continue

            # Applica split train/val come RADCUBE_DATASET_TIME
            arr = np.array(frame_nums)
            if self.mode == "train":
                rem = len(arr) % 30
                arr = arr[:-rem] if rem else arr
                arr = arr.reshape(-1, 30)[:, :27].reshape(-1)
            elif self.mode == "val":
                rem = len(arr) % 30
                arr = arr[:-rem] if rem else arr
                arr = arr.reshape(-1, 30)[:, -3:].reshape(-1)
            elif self.mode == "test":
                rem = len(arr) % self.n_frames
                arr = arr[:-rem] if rem else arr
            frame_nums = arr.tolist()

            # Timestamps
            ts_mat_path = cubes_dir / "timestamps.mat"
            frame_ts: dict[int, int] = {}
            if ts_mat_path.exists():
                mat = scipy.io.loadmat(str(ts_mat_path))["unixDateTime"]
                for idx in frame_nums:
                    if idx <= len(mat):
                        frame_ts[idx] = int(mat[idx - 1][0]) * 10**9

            # LiDAR timestamp → path
            lidar_ts2path: dict[int, str] = {}
            if lidar_dir.exists():
                lidar_ts2path = _get_ts_paths(str(lidar_dir))

            for idx in frame_nums:
                ts_ns = frame_ts.get(idx, idx)
                lidar_path = ""
                if lidar_ts2path:
                    lt = _closest_ts(ts_ns, lidar_ts2path)
                    lidar_path = lidar_ts2path[lt]

                aux.append({
                    "scene":      scene_num,
                    "frame_idx":  idx,
                    "power_path": str(cubes_dir / f"Pow_Frame_{idx}.mat"),
                    "ele_path":   str(cubes_dir / f"Ele_Frame_{idx}.mat"),
                    "lidar_path": lidar_path,
                    "timestamp":  ts_ns,
                })

        # Raggruppa in triplet
        return [aux[i: i + self.n_frames]
                for i in range(0, len(aux), self.n_frames)
                if len(aux[i: i + self.n_frames]) == self.n_frames]

    # ------------------------------------------------------------------
    # Costruzione tensore [R, A, E, 2] da un singolo frame
    # ------------------------------------------------------------------

    def _load_frame_tensor(self, info: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Carica Pow_Frame + Ele_Frame e costruisce:
          tensor_rae2 : (R=500, A=240, E=34, 2)  [intensity, velocity]
          rad_map      : (R=500, A=240, D=128)    [power cube trasposto]

        Costruzione (blue path del paper, Sec 3.2):
          intensity[r,a,e] = power[r, argmax_D(power[r,:,a]), a]  se elevationIndex→e
          velocity[r,a,e]  = (argmax_D − D//2) × vel_bin_size       normalizzato [-1,1]
        """
        # Carica il cubo radar (già processato: Range FFT + Doppler FFT + AoA)
        power = scipy.io.loadmat(info["power_path"])["radarCube"].astype(np.float32)
        ele   = scipy.io.loadmat(info["ele_path"])["elevationIndex"].astype(np.float32)

        # Normalizzazione come nel repo RaDelft
        # REF: RADCUBE_DATASET_TIME.__getitem__
        power = power / _POWER_NORM                              # (R=500, D=128, A=240)
        ele   = np.nan_to_num(ele, nan=17.0) / float(_ELEV_N_BINS)  # (R=500, A=240) ∈[0,1]

        R_raw, D, A_raw = power.shape    # (500, 128, 240)

        # ── RAD map: trasponi [R, D, A] → [R, A, D] ──────────────────
        rad_map = power.transpose(0, 2, 1).copy()               # (500, 240, 128)

        # ── Tensore [R, A, E, 2] ──────────────────────────────────────
        # Vettorizzato: per ogni (r, a) trova il bin Doppler più forte
        d_star    = np.argmax(power, axis=1)                     # (R, A) — bin Doppler
        intensity = power[np.arange(R_raw)[:, None],
                          d_star,
                          np.arange(A_raw)[None, :]]             # (R, A)

        # Velocità in m/s e normalizzazione
        velocity = (d_star.astype(np.float32) - D / 2.0) * _VEL_BIN_SIZE   # (R, A)
        if self.norm_vel:
            velocity = np.clip(velocity / self.vel_max, -1.0, 1.0)

        # Elevation bin fisico: round(ele * (E-1))  ∈ [0, E-1]
        el_bin = np.round(ele * (self._E_max_idx)).astype(np.int32)
        el_bin = np.clip(el_bin, 0, self.E - 1)

        # Popola il tensore
        tensor_rae2 = np.zeros((R_raw, A_raw, self.E, 2), dtype=np.float32)
        r_idx = np.arange(R_raw)[:, None]       # (R, 1) broadcast
        a_idx = np.arange(A_raw)[None, :]       # (1, A) broadcast
        tensor_rae2[r_idx, a_idx, el_bin, 0] = intensity
        tensor_rae2[r_idx, a_idx, el_bin, 1] = velocity

        return tensor_rae2, rad_map

    @property
    def _E_max_idx(self) -> int:
        return self.E - 1

    # ------------------------------------------------------------------
    # Caricamento GT LiDAR (già ground-removed — NON ri-applicare Patchwork++)
    # ------------------------------------------------------------------

    def _load_lidar_gt(self, lidar_path: str) -> np.ndarray:
        """
        Carica il LiDAR pre-processato e lo voxelizza.

        IMPORTANTE: rslidar_points_clean è GIÀ ground-removed!
        REF: data_preparation.py::clean_and_save_lidar che applica
             remove_ground_points_patchwork + filtra z>-2 + rimuove ego-car
             → salvato in rslidar_points_clean PRIMA del training.
        → NON applicare Patchwork++ qui!

        Formato (rs_lidar_clean):
            np.load(path) → (N, 3) float32  [x, y, z] cartesiane

        Calibrazione LiDAR→Radar (da data_preparation.py::prepare_lidar_pointcloud):
            Rotazione di azimuth_offset=7° attorno all'asse z
            + traslazione x_offset/y_offset (qui entrambi = 0)
        """
        if not lidar_path or not os.path.exists(lidar_path):
            return np.zeros((self.R, self.A, self.E), dtype=np.float32)

        # Leggi il punto cloud pulito
        pts = np.load(lidar_path)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts = pts[:, :3].astype(np.float64)

        if len(pts) == 0:
            return np.zeros((self.R, self.A, self.E), dtype=np.float32)

        # Calibrazione: rotazione 7° + traslazione
        # REF: data_preparation.transform_point_cloud(lidar, [0,0,7], [x/100, y/100, 0])
        pts_cal = pts @ self._R_mat.T
        pts_cal[:, 0] += _X_OFFSET_M
        pts_cal[:, 1] += _Y_OFFSET_M

        # Voxelizza: cartesiano → sferico → griglia non-uniforme → (R, A, E)
        return self.voxelizer.voxelize(pts_cal)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        triplet = self.samples[idx]    # [info_t-2, info_t-1, info_t]

        # Carica i 3 frame
        tensors, _ = [], None
        for i, info in enumerate(triplet):
            t, rad = self._load_frame_tensor(info)
            tensors.append(t)
            if i == len(triplet) - 1:
                rad_map_t = rad   # RAD map del frame più recente

        # Fusione temporale: [t, t-1, t-2] lungo l'asse canali → (R, A, E, 6)
        # paper: "current and previous two frames" → canali [0,1]=t, [2,3]=t-1, [4,5]=t-2
        # tensors[0]=t-2, tensors[1]=t-1, tensors[2]=t
        t2, t1, t0 = tensors[0], tensors[1], tensors[2]   # t-2, t-1, t
        radar_cube = np.concatenate([t0, t1, t2], axis=-1)  # (R, A, E, 6)

        # GT occupancy dal frame più recente
        lidar_occ = self._load_lidar_gt(triplet[-1]["lidar_path"])  # (R, A, E)

        # Tensori PyTorch channel-first
        sample = {
            "radar_cube": torch.from_numpy(
                radar_cube.transpose(3, 0, 1, 2)    # (6, R, A, E)
            ),
            "rad_map": torch.from_numpy(
                rad_map_t.transpose(2, 0, 1)         # (D, R, A)
            ),
            "lidar_occ": torch.from_numpy(lidar_occ),   # (R, A, E)
            "meta": {
                "scene":     triplet[-1]["scene"],
                "frame_idx": triplet[-1]["frame_idx"],
                "mode":      self.mode,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
