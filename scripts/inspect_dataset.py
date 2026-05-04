"""
Ispeziona la struttura e le shape dei file di un dataset radar.
Esegui dalla root del tuo dataset:

    python /path/to/RadarMamba/scripts/inspect_dataset.py --path /path/to/dataset

Oppure passando un file specifico:
    python scripts/inspect_dataset.py --path /path/to/dataset --max_depth 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def print_tree(root: Path, max_depth: int = 3, _depth: int = 0, _max_files: int = 5):
    """Stampa la struttura ad albero della directory."""
    if _depth > max_depth:
        return
    try:
        items = sorted(root.iterdir())
    except PermissionError:
        return

    dirs  = [x for x in items if x.is_dir()]
    files = [x for x in items if x.is_file()]

    indent = "  " * _depth

    for d in dirs:
        n_children = sum(1 for _ in d.iterdir()) if d.exists() else 0
        print(f"{indent}📁 {d.name}/  ({n_children} files)")
        print_tree(d, max_depth, _depth + 1, _max_files)

    shown = 0
    for f in files:
        if shown >= _max_files:
            print(f"{indent}... e altri {len(files)-shown} file")
            break
        size_kb = f.stat().st_size / 1024
        print(f"{indent}📄 {f.name}  ({size_kb:.0f} KB)")
        shown += 1


def inspect_file(path: Path):
    """Ispeziona la shape di un file numpy/mat."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            import numpy as np
            arr = np.load(str(path), allow_pickle=True)
            if isinstance(arr, np.ndarray):
                print(f"  shape={arr.shape}  dtype={arr.dtype}")
            else:
                # allow_pickle può restituire un oggetto
                print(f"  tipo={type(arr)}  keys={list(arr.item().keys()) if hasattr(arr, 'item') else '?'}")

        elif suffix == ".npz":
            import numpy as np
            d = np.load(str(path))
            for k in d.files:
                print(f"  [{k}]: shape={d[k].shape}  dtype={d[k].dtype}")

        elif suffix == ".mat":
            import scipy.io
            mat = scipy.io.loadmat(str(path))
            keys = [k for k in mat.keys() if not k.startswith("__")]
            for k in keys:
                v = mat[k]
                if hasattr(v, "shape"):
                    print(f"  [{k}]: shape={v.shape}  dtype={v.dtype}")
                else:
                    print(f"  [{k}]: {type(v)}")

        elif suffix == ".bin":
            import numpy as np
            # Prova a leggere come int16 (formato ADC RADIal)
            raw = np.fromfile(str(path), dtype=np.int16)
            print(f"  int16 → n_elements={len(raw)}  "
                  f"(se ADC RADIal: {len(raw)} = samples×rx×chirps×2?)")

        elif suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(str(path), nrows=3)
            print(f"  CSV: {len(df.columns)} colonne, prime colonne: {list(df.columns[:6])}")

        elif suffix in (".jpg", ".jpeg", ".png", ".avi"):
            print(f"  (immagine/video — non ispezioniamo)")

        else:
            print(f"  (formato non riconosciuto)")

    except Exception as e:
        print(f"  Errore lettura: {e}")


def auto_inspect(root: Path, max_files_per_ext: int = 2):
    """
    Trova automaticamente file radar/lidar e ne ispeziona la shape.
    """
    extensions = {".npy", ".npz", ".mat", ".bin"}
    found: dict[str, list[Path]] = {}

    for dirpath, _, filenames in os.walk(str(root)):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in extensions:
                key = p.suffix.lower()
                found.setdefault(key, []).append(p)

    print("\n" + "=" * 60)
    print("FILE TROVATI (shape)")
    print("=" * 60)

    for ext, paths in found.items():
        print(f"\n── {ext.upper()} ({len(paths)} file totali) ──")
        for p in paths[:max_files_per_ext]:
            print(f"  {p.relative_to(root)}")
            inspect_file(p)
        if len(paths) > max_files_per_ext:
            print(f"  ... e altri {len(paths) - max_files_per_ext} file {ext}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",      required=True, help="Path al dataset")
    parser.add_argument("--max_depth", type=int, default=4, help="Profondità albero cartelle")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Path non trovato: {root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"STRUTTURA DATASET: {root.name}")
    print(f"{'='*60}")
    print_tree(root, max_depth=args.max_depth)

    auto_inspect(root, max_files_per_ext=2)

    print("\n" + "=" * 60)
    print("RIEPILOGO — cosa devi comunicare:")
    print("=" * 60)
    print("  1. Nome del dataset (RADIal / RaDelft / altro?)")
    print("  2. Shape dei file FFT/radar")
    print("  3. Shape dei file LiDAR")
    print("  4. Presenza di file .mat con 'radarCube' o 'elevationIndex'?")
    print("  5. Presenza di file ADC .bin?")


if __name__ == "__main__":
    main()
