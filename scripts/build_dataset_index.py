"""
Build Dataset Index
====================
Scansiona le cartelle raw del dataset e genera l'indice JSON
con la lista di tutti i campioni (scene, frame, path, split).

L'indice è usato da RADIalDataset e RaDelftDataset per caricare i dati.

Uso:
    python scripts/build_dataset_index.py --config configs/radial.yaml
    python scripts/build_dataset_index.py --config configs/radelft.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Aggiungi la root al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf


def build_radial_index(cfg) -> list[dict]:
    """
    Scansiona la directory RADIal e costruisce l'indice.

    REF: https://github.com/valeoai/RADIal
    RADIal ha 91 sequenze. Per ogni sequenza, ogni frame è indicizzato
    tramite il log.txt che fornisce i timestamp per ogni modality.

    Returns:
        lista di dict, uno per campione (frame).

    TODO:
        1. Trova tutte le sequenze in cfg.dataset.raw_path:
               raw_path = Path(cfg.dataset.raw_path)
               sequences = sorted(raw_path.glob("RADIal_sequence_*"))

        2. Per ogni sequenza, leggi il log.txt per trovare i frame sincronizzati:
               with open(seq / "log.txt") as f: ...
               (il log.txt ha timestamp per ogni modality per ogni frame)

        3. Per ogni frame t (con t >= 2 per avere t-2 disponibile):
               sample = {
                   "scene_id":    seq.name,
                   "frame_idx":   t,
                   "frame_paths": [
                       str(seq / f"frame_{t-2:06d}"),
                       str(seq / f"frame_{t-1:06d}"),
                       str(seq / f"frame_{t:06d}"),
                   ],
                   "lidar_path": str(seq / f"laser_PCL_{t:06d}.bin"),
                   "split":      "test" if seq.name in test_seqs else "train"
               }
               samples.append(sample)

        4. Nota: la struttura esatta delle cartelle dipende da come RADIal
           organizza i frame. Controlla il repo ufficiale e il DBReader.

        5. Split test: cfg.dataset.split.test_sequences (lista di nomi)
           Se non configurata, usa gli ultimi 10 (come nel paper).
    """
    raise NotImplementedError(
        "TODO: implementa build_radial_index.\n"
        "Hint: esplora data/raw/radial/ con ls e ispeziona il formato log.txt."
    )


def build_radelft_index(cfg) -> list[dict]:
    """
    Scansiona le scene RaDelft e costruisce l'indice.

    REF: https://github.com/RaDelft/RaDelft-Dataset
    RaDelft ha 7 scene. Scene 2 e 6 = test, resto = train.

    Returns:
        lista di dict, uno per campione.

    TODO:
        1. Per ogni scena in cfg.dataset.split.train_scenes + test_scenes:
               scene_path = Path(cfg.dataset.raw_path) / f"scene_{scene_id}"

        2. Trova i frame disponibili nella cartella radar_cube/:
               cube_files = sorted((scene_path / "radar_cube").glob("*.npy"))
               # oppure "*.npz" o "*.mat" — controlla la versione del dataset

        3. Per ogni frame t (con t >= 2):
               sample = {
                   "scene_id":    scene_id,
                   "frame_idx":   t,
                   "frame_paths": [
                       str(cube_files[t-2]),
                       str(cube_files[t-1]),
                       str(cube_files[t]),
                   ],
                   "lidar_path": str(scene_path / "lidar" / cube_files[t].stem + ".bin"),
                   "split":      "test" if scene_id in [2, 6] else "train"
               }
               samples.append(sample)
    """
    raise NotImplementedError(
        "TODO: implementa build_radelft_index.\n"
        "Hint: esplora data/raw/radelft/scene_1/ e controlla i formati dei file."
    )


def main():
    parser = argparse.ArgumentParser(description="Build dataset index JSON")
    parser.add_argument("--config", required=True, help="Path al config YAML")
    parser.add_argument("--output_dir", default=None,
                        help="Directory di output (default: processed_path dal config)")
    args = parser.parse_args()

    # Carica config
    cfg = OmegaConf.load(args.config)
    dataset_name = cfg.dataset.name

    print(f"Building index for dataset: {dataset_name}")
    print(f"Raw data path: {cfg.dataset.raw_path}")

    # Costruisci indice
    if dataset_name == "radial":
        samples = build_radial_index(cfg)
    elif dataset_name == "radelft":
        samples = build_radelft_index(cfg)
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}")

    # Split in train/test
    train_samples = [s for s in samples if s["split"] == "train"]
    test_samples  = [s for s in samples if s["split"] == "test"]

    print(f"Train: {len(train_samples)} samples")
    print(f"Test:  {len(test_samples)} samples")

    # Salva
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.dataset.processed_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "index_train.json"
    test_path  = out_dir / "index_test.json"

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=2)

    print(f"Salvato: {train_path}")
    print(f"Salvato: {test_path}")


if __name__ == "__main__":
    main()
