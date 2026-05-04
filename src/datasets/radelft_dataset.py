"""
RaDelft Dataset — PyTorch Dataset (versione temporale, 3 frame)
================================================================
REF: https://github.com/RaDelft/RaDelft-Dataset
     machine_learning_python/loaders/rad_cube_loader.py::RADCUBE_DATASET_TIME

Struttura attesa sul disco (dall'analisi della repo ufficiale):
    data/raw/radelft/
    └── Scene{N}/
        ├── RadarCubes/
        │   ├── Pow_Frame_{idx}.mat       → radarCube: [R=500, D=128, A=240] float32
        │   ├── Ele_Frame_{idx}.mat       → elevationIndex: [R=500, A=240] float32
        │   ├── DopFold_Frame_{idx}.mat   → velocità Doppler unfolded (opzionale)
        │   └── timestamps.mat            → frame_num_to_timestamp["unixDateTime"]
        └── rosDS/
            ├── rslidar_points_clean/
            │   └── {timestamp_sec}.{ns}.npy   → (N, 3) float32 [x, y, z]  (GiÀ GROUND-REMOVED)
            └── ueye_left_image_rect_color/     (immagini camera, non usate)

Formato input per Radar-Mamba:
    - radar_cube: (6, R, A, E) float32   ← 3 frame × [power, elevation_idx]
    - rad_map:    (D, R, A)   float32   ← RAD map del frame corrente (t)
    - lidar_occ:  (R, A, E)   float32   ← GT occupancy dal LiDAR (solo per il frame t)

Costruzione del tensore [R, A, E, 2] dal cube RaDelft:
    power:     Pow_Frame.mat["radarCube"]         [R=500, D=128, A=240], / 8998.5576
    elevation: Ele_Frame.mat["elevationIndex"]    [R=500, A=240], / 34  (normalizzato 0-1)

    Per ogni cella (r, a):
        el_bin     = round(elevation[r, a] * E)      ← bin di elevation fisico
        d_star     = argmax(power[r, :, a])           ← Doppler bin a potenza max
        intensity  = power[r, d_star, a]
        velocity   = (d_star - D//2) * vel_bin_size   ← in m/s
        → tensor_rae2[r, a, el_bin, 0] = intensity (normalizzata)
        → tensor_rae2[r, a, el_bin, 1] = velocity (normalizzata in [-1,1])

    RAD map: rad_map[r, a, d] = power[r, d, a]  (solo transpose)

Split:
    Train: Scene 1, 3, 4, 5, 7   (90% di ogni scena)
    Val:   Scene 1, 3, 4, 5, 7   (10% di ogni scena, ogni 10 frame)
    Test:  Scene 2, 6             (tutti i frame)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from src.alignment.voxelization import Voxelizer, radelft_default_axes
from src.alignment.coordinate_transform import cartesian_to_spherical_rad


# Parametri RaDelft da data_preparation.py::get_default_params()
_VEL_FFT_SIZE  = 128
_VEL_BIN_SIZE  = 0.04607058455831936   # m/s per bin Doppler
_POWER_NORM    = 8998.5576
_ELEV_NORM     = 34.0                  # numero bin elevation (normalizzazione)
_AZIMUTH_OFFSET_DEG = 7.0             # rotazione tra LiDAR e radar (gradi, asse z)
_X_OFFSET_CM   = 0.0
_Y_OFFSET_CM   = 0.0


def _get_timestamps_and_paths(directory: str) -> dict[int, str]:
    """
    Replica data_preparation.get_timestamps_and_paths.
    Restituisce {timestamp_ns: filepath} per tutti i .npy in directory.
    """
    out = {}
    for fname in os.listdir(directory):
        if fname.endswith(".npy") or fname.endswith(".jpg") or fname.endswith(".mat"):
            parts = fname.split(".")
            if len(parts) >= 2:
                try:
                    ts = int(parts[0]) * 10**9 + int(parts[1])
                    out[ts] = os.path.join(directory, fname)
                except ValueError:
                    pass
    return out


def _closest_timestamp(new_ts: int, ts_dict: dict[int, str]) -> int:
    """Replica data_preparation.closest_timestamp."""
    return min(ts_dict.keys(), key=lambda t: abs(t - new_ts))


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """Matrice di rotazione 3D attorno all'asse z."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1],
    ])


class RaDelftDataset(Dataset):
    """
    Dataset PyTorch per RaDelft — versione temporale (3 frame consecutivi).
    Replica la logica di RADCUBE_DATASET_TIME dal repo ufficiale.
    """

    # Split ufficiali (paper Radar-Mamba Section 4.1 + repo RaDelft)
    TRAIN_SCENES = [1, 3, 4, 5, 7]
    TEST_SCENES  = [2, 6]

    def __init__(
        self,
        dataset_path: str | Path,
        mode: str = "train",         # "train" | "val" | "test"
        n_frames: int = 3,           # finestra temporale (paper: 3)
        normalize_velocity: bool = True,
        vel_max_mps: float = 5.0,    # clipping velocità per normalizzazione
        transform=None,
    ) -> None:
        """
        Args:
            dataset_path: root del dataset (es. data/raw/radelft/).
            mode:         split.
            n_frames:     numero di frame per finestra temporale (default 3).
            normalize_velocity: normalizza Doppler in [-1, 1].
            vel_max_mps:  massima velocità per normalizzazione.
        """
        assert mode in ("train", "val", "test"), f"mode deve essere train/val/test, ricevuto: {mode}"

        self.root        = Path(dataset_path)
        self.mode        = mode
        self.n_frames    = n_frames
        self.norm_vel    = normalize_velocity
        self.vel_max     = vel_max_mps
        self.transform   = transform

        # Assi fisici RaDelft (non-uniformi)
        self.range_axis, self.azimuth_axis, self.elevation_axis = radelft_default_axes()
        self.E = len(self.elevation_axis)   # 34
        self.R = len(self.range_axis)       # ~487
        self.A = len(self.azimuth_axis)     # 240
        self.D = _VEL_FFT_SIZE              # 128

        # Voxelizzatore per il LiDAR
        self.voxelizer = Voxelizer(self.range_axis, self.azimuth_axis, self.elevation_axis)

        # Costruisci l'indice dei campioni
        scenes = self.TRAIN_SCENES if mode in ("train", "val") else self.TEST_SCENES
        self.samples = self._build_index(scenes)

    # ------------------------------------------------------------------
    # Costruzione dell'indice
    # ------------------------------------------------------------------

    def _build_index(self, scenes: list[int]) -> list[dict]:
        """
        Replica la logica di RADCUBE_DATASET_TIME.__init__ per costruire
        i gruppi di 3 frame consecutivi.
        """
        aux_frames  = []   # lista di dict per frame singoli

        for scene_num in scenes:
            scene_dir = self.root / f"Scene{scene_num}"
            cubes_dir = scene_dir / "RadarCubes"
            rods_dir  = scene_dir / "rosDS"
            lidar_dir = rods_dir / "rslidar_points_clean"

            if not cubes_dir.exists():
                raise FileNotFoundError(f"RadarCubes non trovata: {cubes_dir}")

            # Trova tutti i Pow_Frame_X.mat
            pow_files = [f for f in os.listdir(cubes_dir) if "Pow_Frame" in f]
            frame_nums = sorted([int(f.split("_")[-1].split(".")[0]) for f in pow_files])

            # Split train/val: 9 train + 1 val ogni 10 frame
            if self.mode == "train":
                rem = len(frame_nums) % 30
                arr = np.array(frame_nums)
                if rem != 0:
                    arr = arr[:-rem].reshape(-1, 30)[:, :27].reshape(-1)
                else:
                    arr = arr.reshape(-1, 30)[:, :27].reshape(-1)
                frame_nums = arr.tolist()
            elif self.mode == "val":
                rem = len(frame_nums) % 30
                arr = np.array(frame_nums)
                if rem != 0:
                    arr = arr[:-rem].reshape(-1, 30)[:, -3:].reshape(-1)
                else:
                    arr = arr.reshape(-1, 30)[:, -3:].reshape(-1)
                frame_nums = arr.tolist()
            # "test": usa tutti i frame, ma mantieni divisibile per n_frames
            elif self.mode == "test":
                rem = len(frame_nums) % self.n_frames
                if rem != 0:
                    frame_nums = frame_nums[:-rem]

            # Carica timestamp
            ts_mat = scipy.io.loadmat(str(cubes_dir / "timestamps.mat"))
            frame_num_to_ts = ts_mat["unixDateTime"]

            # Carica mapping timestamp → path LiDAR
            lidar_ts2path = _get_timestamps_and_paths(str(lidar_dir))

            for idx in frame_nums:
                ts_ns = int(frame_num_to_ts[idx - 1][0]) * 10**9

                # Path più vicino nel LiDAR
                lidar_ts  = _closest_timestamp(ts_ns, lidar_ts2path)
                lidar_path = lidar_ts2path[lidar_ts]

                aux_frames.append({
                    "scene":      scene_num,
                    "frame_idx":  idx,
                    "power_path": str(cubes_dir / f"Pow_Frame_{idx}.mat"),
                    "ele_path":   str(cubes_dir / f"Ele_Frame_{idx}.mat"),
                    "lidar_path": lidar_path,
                    "timestamp":  ts_ns,
                })

        # Raggruppa in triplet di frame consecutivi
        samples = []
        for i in range(0, len(aux_frames), self.n_frames):
            triplet = aux_frames[i: i + self.n_frames]
            if len(triplet) == self.n_frames:
                samples.append(triplet)

        return samples

    # ------------------------------------------------------------------
    # Costruzione del tensore [R, A, E, 2] da un singolo frame
    # ------------------------------------------------------------------

    def _load_single_frame(self, frame_info: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Carica un frame RaDelft e costruisce:
          tensor_rae2: (R, A, E, 2) — [normalized_intensity, normalized_velocity]
          rad_map:     (R, A, D)    — power cube trasposto

        Segue la logica di RADCUBE_DATASET_TIME.__getitem__:
          power     = radarCube / 8998.5576
          elevation = elevationIndex / 34   (indice normalizzato ∈ [0, 1])
        """
        # Carica i file .mat
        power_raw = scipy.io.loadmat(frame_info["power_path"])["radarCube"].astype(np.float32)
        ele_raw   = scipy.io.loadmat(frame_info["ele_path"])["elevationIndex"].astype(np.float32)

        # Normalizzazione come nel repo RaDelft
        power = power_raw / _POWER_NORM       # (R, D, A)
        # NaN elevation → imposta a 17.0 (indice medio) poi normalizza
        ele   = np.nan_to_num(ele_raw, nan=17.0) / _ELEV_NORM   # (R, A), ∈ [0, 1]

        R_raw, D, A_raw = power.shape

        # ── RAD map ────────────────────────────────────────────────────
        # rad_map[r, a, d] = power[r, d, a]
        rad_map = power.transpose(0, 2, 1).astype(np.float32)   # (R, A, D)

        # ── Tensore [R, A, E, 2] ───────────────────────────────────────
        # Per ogni cella (r, a):
        #   el_bin  = round(ele[r, a] * E)   → bin fisico elevation
        #   d_star  = argmax(power[r, :, a]) → bin Doppler a max potenza
        #   intensity = power[r, d_star, a]  (già normalizzata)
        #   velocity  = (d_star - D//2) * vel_bin_size  → in m/s

        tensor_rae2 = np.zeros((R_raw, A_raw, self.E, 2), dtype=np.float32)

        # Vettorizzato: D_star per ogni (r, a)
        d_star    = np.argmax(power, axis=1)                     # (R, A)
        intensity = power[np.arange(R_raw)[:, None],
                          d_star,
                          np.arange(A_raw)[None, :]]              # (R, A)

        velocity  = (d_star.astype(np.float32) - D // 2) * _VEL_BIN_SIZE   # (R, A) m/s

        # Normalizza velocità in [-1, 1]
        if self.norm_vel:
            velocity = np.clip(velocity / self.vel_max, -1.0, 1.0)

        # Mappa elevation bin: el_bin ∈ [0, E-1]
        el_bin = np.round(ele * (self.E - 1)).astype(np.int32)
        el_bin = np.clip(el_bin, 0, self.E - 1)

        # Assegna i valori ai voxel
        r_idx = np.arange(R_raw)[:, None]   # (R, 1)
        a_idx = np.arange(A_raw)[None, :]   # (1, A)
        tensor_rae2[r_idx, a_idx, el_bin, 0] = intensity   # intensità
        tensor_rae2[r_idx, a_idx, el_bin, 1] = velocity    # Doppler

        return tensor_rae2, rad_map

    # ------------------------------------------------------------------
    # Caricamento LiDAR e GT occupancy
    # ------------------------------------------------------------------

    def _load_lidar_gt(self, lidar_path: str) -> np.ndarray:
        """
        Carica il LiDAR pre-processato (già ground-removed, formato rs_lidar_clean)
        e lo voxelizza per ottenere la GT occupancy [R, A, E].

        Applica calibrazione radar-LiDAR (rotazione + traslazione) come in
        data_preparation::prepare_lidar_pointcloud.
        """
        pts = np.load(lidar_path).reshape(-1, 3).astype(np.float64)  # (N, 3) xyz

        # Calibrazione: rotazione attorno z di azimuth_offset + traslazione
        # REF: data_preparation.transform_point_cloud con azimuth_offset=7°
        R_mat = _rotation_matrix_z(_AZIMUTH_OFFSET_DEG)
        pts_rot = pts[:, :3] @ R_mat.T
        pts_rot[:, 0] += _X_OFFSET_CM / 100.0
        pts_rot[:, 1] += _Y_OFFSET_CM / 100.0

        # Voxelizza: Cartesiano → sferico → griglia non-uniforme → (R, A, E)
        occ = self.voxelizer.voxelize(pts_rot)

        # Crop alla dimensione attesa (R_raw potrebbe essere 487, R axis lo dice)
        occ = occ[:len(self.range_axis), :len(self.azimuth_axis), :]
        return occ.astype(np.float32)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        triplet = self.samples[idx]   # lista di n_frames frame_info

        # Carica i n_frames tensori [R, A, E, 2]
        frame_tensors = []
        for frame_info in triplet:
            t_rae2, rad_map = self._load_single_frame(frame_info)
            frame_tensors.append(t_rae2)

        # Fusione temporale: concatena lungo l'asse canali → (R, A, E, 6)
        # Ordine: [t-2, t-1, t] → canali [0,1]=t-2, [2,3]=t-1, [4,5]=t
        # (la convexione è [t, t-1, t-2] nel paper Section 3.2)
        padding = np.zeros_like(frame_tensors[0])
        while len(frame_tensors) < 3:
            frame_tensors.insert(0, padding.copy())

        # frame[0]=t-2, frame[1]=t-1, frame[2]=t (frame più recente = ultimo)
        radar_cube = np.concatenate(
            [frame_tensors[2], frame_tensors[1], frame_tensors[0]], axis=-1
        )  # (R, A, E, 6): canali [I_t, D_t, I_{t-1}, D_{t-1}, I_{t-2}, D_{t-2}]

        # RAD map del frame più recente t
        _, rad_map_t = self._load_single_frame(triplet[-1])   # (R, A, D)

        # GT occupancy dal frame più recente t
        lidar_occ = self._load_lidar_gt(triplet[-1]["lidar_path"])   # (R, A, E)

        # Converti in tensori PyTorch con shape channel-first
        radar_cube_t = torch.from_numpy(
            radar_cube.transpose(3, 0, 1, 2)   # (6, R, A, E)
        )
        rad_map_t_tensor = torch.from_numpy(
            rad_map_t.transpose(2, 0, 1)        # (D, R, A)
        )
        lidar_occ_t = torch.from_numpy(lidar_occ)   # (R, A, E)

        sample = {
            "radar_cube": radar_cube_t,       # (6, R, A, E)
            "rad_map":    rad_map_t_tensor,    # (D, R, A)
            "lidar_occ":  lidar_occ_t,         # (R, A, E)
            "meta": {
                "scene":     triplet[-1]["scene"],
                "frame_idx": triplet[-1]["frame_idx"],
                "mode":      self.mode,
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
