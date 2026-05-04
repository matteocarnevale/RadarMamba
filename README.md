# Radar-Mamba

Reimplementazione di **Radar-Mamba: 4D Millimeter-Wave Point Cloud Enhancement via State Space Models**  
(Gao et al., ACM MM 2025)

---

## Architettura (schema del paper)

```
Raw ADC data ──► FFT / DoA ──► Radar tensor [R,A,E,6] ──► EDC Encoder ──► RM Blocks ──► Decoder ──► Occupancy [R,A,E]
                               RAD map [R,A,D] ──────────────────────────► Doppler Backbone + DEF Block ┘
LiDAR ──► Ground removal ──► Polar transform ──► KD-filter ──► FoV clip ──► Voxelize ──► GT [R,A,E]
```

---

## Dataset

### RADIal
- **Paper:** Rebut et al., CVPR 2022  
- **Repo:** https://github.com/valeoai/RADIal  
- **Download:** https://github.com/valeoai/RADIal (link nella sezione Download)
- **Struttura attesa dopo il download:**
  ```
  data/raw/radial/
  ├── RADIal_sequence_000/
  │   ├── adc_data_00.bin   # chip 0 ADC raw
  │   ├── adc_data_01.bin   # chip 1 ADC raw
  │   ├── adc_data_02.bin   # chip 2 ADC raw
  │   ├── adc_data_03.bin   # chip 3 ADC raw
  │   ├── laser_PCL.bin     # LiDAR point cloud
  │   ├── camera.avi        # MJPEG camera
  │   ├── gps.txt
  │   ├── can.bin
  │   └── log.txt           # timestamp per modality
  ├── RADIal_sequence_001/
  ...
  └── labels.csv            # bounding-box labels (veicoli)
  ```
- **Grid (paper Sec 4.1):** [R, A, E] = [480, 736, 11],  
  range [0,50]m, azimuth [−75°,75°], elevation [−4°,6°]

### RaDelft
- **Paper:** Roldan et al., IEEE TRS 2024  
- **Repo:** https://github.com/RaDelft/RaDelft-Dataset  
- **Download:** richiedi accesso su 4TU.ResearchData (link nel repo ufficiale) — limitato a ricercatori accademici
- **Struttura attesa dopo il download:**
  ```
  data/raw/radelft/
  ├── scene_1/
  │   ├── radar_cube/       # file .npy o .mat per ogni frame
  │   ├── lidar/            # file .bin o .npy per ogni frame
  │   ├── camera/
  │   ├── odometry/
  │   └── calibration.json
  ├── scene_2/
  ...
  └── scene_7/
  ```
  > Nota: la struttura esatta dipende dalla versione del dataset (attuale: v5).  
  > Consulta i notebook del repo ufficiale (`1_frame_loader`) per i path esatti.
- **Grid (paper Sec 4.1):** [R, A, E] = [512, 256, 34],  
  range [0,50]m, azimuth [−70°,70°], elevation [−15°,15°]

---

## Setup

```bash
# 1. Crea ambiente virtuale
python -m venv .venv && source .venv/bin/activate

# 2. Installa dipendenze base
pip install -r requirements.txt

# 3. Installa Mamba (richiede CUDA >= 11.6)
pip install mamba-ssm causal-conv1d

# 4. Installa il pacchetto in modalità sviluppo
pip install -e .
```

---

## Workflow

### 1. Preprocessing — debug su una singola sequenza

```bash
python scripts/preprocess_one_scene.py \
    --config configs/radial.yaml \
    --scene_path data/raw/radial/RADIal_sequence_000 \
    --out_dir /tmp/debug_preprocess
```

Salva `radar_cube.npy`, `rad_map.npy`, `lidar_occ.npy` e visualizzazioni BEV.

### 2. Costruire l'indice del dataset

```bash
python scripts/build_dataset_index.py --config configs/radial.yaml
# oppure
python scripts/build_dataset_index.py --config configs/radelft.yaml
```

Genera `data/processed/<dataset>/index_train.json` e `index_test.json`.

### 3. Preprocessing completo (opzionale — salva .npz)

```bash
# Viene lanciato automaticamente dal Dataset se processed_path è configurato
python scripts/preprocess_all.py --config configs/radial.yaml
```

### 4. Training

```bash
python scripts/train.py --config configs/radial.yaml
```

### 5. Valutazione

```bash
python scripts/eval.py \
    --config configs/radial.yaml \
    --checkpoint checkpoints/best.pth
```

---

## Struttura del repository

```
radar_mamba/
├── configs/           # YAML: parametri dataset, modello, training
├── data/
│   ├── raw/           # symlink o dati scaricati (non versionare GB)
│   └── processed/     # tensori .npz preprocessati (cache opzionale)
├── src/
│   ├── calibration/   # caricamento calibrazione extrinsic radar–lidar
│   ├── alignment/     # 5 passi cross-modal (paper Sec 3.2)
│   │   ├── ground_removal.py       # Patchwork++
│   │   ├── coordinate_transform.py # Cartesiano → polare
│   │   ├── radar_lidar_filter.py   # KD-tree + soglia 0.5 m (Eq. 1-2)
│   │   ├── fov_alignment.py        # ritaglio Field-of-View
│   │   ├── voxelization.py         # griglia occupancy [R,A,E]
│   │   └── pipeline.py             # orchestratore dei 5 passi
│   ├── radar_tensor/  # ADC → tensore 4D [R,A,E,6] + mappa RAD
│   ├── datasets/      # PyTorch Dataset (RADIal, RaDelft)
│   ├── models/        # U-Net, RM block, RHSS, DEF, CGSA, decoder
│   ├── losses/        # Focal Loss (α=0.995, γ=2)
│   └── utils/         # metriche CD/HD, visualizzazione, CFAR 2D
├── scripts/           # preprocess, train, eval
├── tests/             # unit test su voxel / shape / metriche
└── notebooks/         # esplorazione dataset (da completare)
```

---

## Citazione

```bibtex
@inproceedings{gao2025radarmamba,
  title     = {Radar-Mamba: 4D Millimeter-Wave Point Cloud Enhancement via State Space Models},
  author    = {Gao, Hong and Xu, Xiangkai and Zhu, Tianqi and Dong, Xiugang and Bao, Yiming and Zhang, Min-Ling},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year      = {2025},
  doi       = {10.1145/3746027.3755431}
}
```
