"""
Reprocess bad RADIal samples (from validate_processed_data.py output).

This script reads a `bad_samples.json` produced by `scripts/validate_processed_data.py`
and regenerates ONLY those `.npz` files (with overwrite), so missing keys like
`rad_map` / `lidar_occ` get fixed without reprocessing the entire dataset.

Usage:
    python scripts/reprocess_bad_samples.py --config configs/radial.yaml \
        --bad_json /media/data/matteo-carnevale/dataset/RADIal_processed/bad_samples.json \
        --workers 8

Then re-run:
    python scripts/validate_processed_data.py --config configs/radial.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.preprocess_radial import process_one_sample


def _extract_sample_id(npz_path: str) -> int | None:
    try:
        stem = Path(npz_path).stem  # "011459"
        return int(stem)
    except Exception:
        return None


def _load_bad_ids(bad_json: Path) -> list[int]:
    data = json.loads(bad_json.read_text())
    ids: list[int] = []
    for entry in data:
        sid = _extract_sample_id(entry.get("path", ""))
        if sid is not None:
            ids.append(sid)
    return sorted(set(ids))


def main() -> None:
    p = argparse.ArgumentParser(description="Reprocess only bad RADIal .npz samples")
    p.add_argument("--config", required=True, help="configs/radial.yaml")
    p.add_argument("--bad_json", required=True, help="Path to bad_samples.json")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    raw_path = Path(cfg.dataset.raw_path)
    proc_path = Path(cfg.dataset.processed_path)
    calib_path = str(cfg.dataset.get("calib_table_path", raw_path / "CalibrationTable.npy"))
    project_root = str(Path(__file__).resolve().parents[1])

    bad_json = Path(args.bad_json)
    if not bad_json.exists():
        raise FileNotFoundError(f"bad_json not found: {bad_json}")

    bad_ids = _load_bad_ids(bad_json)
    if not bad_ids:
        print("No bad ids found in bad_json. Nothing to do.")
        return

    # Build jobs with the same structure used by preprocess_radial.py
    # (we only filter sample_ids; everything else is identical)
    jobs = []
    for sid in bad_ids:
        fft_path = str(raw_path / "radar_FFT" / f"fft_{sid:06d}.npy")
        lidar_path = str(raw_path / "laser_PCL" / f"pcl_{sid:06d}.npy")
        out_path = str(proc_path / f"{sid:06d}.npz")

        if not Path(fft_path).exists():
            continue

        jobs.append(
            {
                "sample_id": sid,
                "fft_path": fft_path,
                "lidar_path": lidar_path,
                "out_path": out_path,
                "calib_path": calib_path,
                "overwrite": True,
                "project_root": project_root,
            }
        )

    print(f"Bad IDs in JSON: {len(bad_ids)}")
    print(f"Jobs to reprocess: {len(jobs)}")

    if not jobs:
        print("No jobs to process. Check raw_path / radar_FFT.")
        return

    # Run with the same multiprocessing pattern used by preprocess_radial.py
    results = {"ok": 0, "skipped": 0, "error": 0}
    errors: list[tuple[int, str]] = []

    if args.workers <= 1:
        for job in tqdm(jobs, desc="Reprocessing"):
            sid, ok, msg = process_one_sample(job)
            if msg == "skipped":
                results["skipped"] += 1
            elif ok:
                results["ok"] += 1
            else:
                results["error"] += 1
                errors.append((sid, msg))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_sample, job): job for job in jobs}
            with tqdm(total=len(jobs), desc="Reprocessing") as pbar:
                for fut in as_completed(futures):
                    sid, ok, msg = fut.result()
                    if msg == "skipped":
                        results["skipped"] += 1
                    elif ok:
                        results["ok"] += 1
                    else:
                        results["error"] += 1
                        errors.append((sid, msg))
                    pbar.update(1)

    print(f"Done. results={results}")
    if errors:
        print("First errors:")
        for sid, msg in errors[:5]:
            print(f"  ✗ {sid:06d}: {msg[:200]}")


if __name__ == "__main__":
    main()

