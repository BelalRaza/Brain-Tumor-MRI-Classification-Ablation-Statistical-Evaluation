#!/usr/bin/env python3
"""Build patient-grouped manifests and pack train/val/test into one .npz for SageMaker / local M4 runs.

Output: data/manifests/*.csv (+ .sha256), data/m4/m4_data.npz, data/m4/split_counts.json
Images are stored single-channel uint8 (all slices are grayscale replicated to 3 channels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, DATA_DIR, IMAGE_SIZE
from src.brain_tumor.repro import sha256_file
from src.brain_tumor.splits import build_manifests, split_counts, write_manifests


def load_gray(rows: list[dict], desc: str) -> np.ndarray:
    out = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, r in enumerate(tqdm(rows, desc=desc)):
        im = cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise IOError(r["file"])
        if im.shape != (IMAGE_SIZE, IMAGE_SIZE):
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        out[i] = im
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "m4" / "m4_data.npz")
    args = ap.parse_args()

    splits = build_manifests(val_frac=args.val_frac, seed=args.seed)
    hashes = write_manifests(splits)
    counts = split_counts(splits)
    print(json.dumps(counts, indent=2))

    train_counts_preaug = [counts["train_source"][c]["images"] for c in CLASS_NAMES]
    payload = {"class_names": np.array(CLASS_NAMES), "train_counts_preaug": np.array(train_counts_preaug)}
    for name in ("train", "val", "test"):
        rows = splits[name]
        payload[f"x_{name}"] = load_gray(rows, name)
        payload[f"y_{name}"] = np.array([r["label"] for r in rows], dtype=np.int8)
        payload[f"pid_{name}"] = np.array([r["patient_id"] for r in rows])
        payload[f"file_{name}"] = np.array([Path(r["file"]).name for r in rows])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    info = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "counts": counts,
        "train_counts_preaug": train_counts_preaug,
        "manifest_sha256": hashes,
        "npz": str(args.out),
        "npz_sha256": sha256_file(args.out),
    }
    (args.out.parent / "split_counts.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print("pre-augmentation train counts:", dict(zip(CLASS_NAMES, train_counts_preaug)))


if __name__ == "__main__":
    main()
