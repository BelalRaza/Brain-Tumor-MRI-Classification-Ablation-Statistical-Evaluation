#!/usr/bin/env python3
"""Prepare BRISC 2025 external validation set (H4): 3 classes, crop+240px, pHash dedupe vs Figshare.

Reads ``brisc2025/classification_task/test/{glioma,meningioma,pituitary}/`` (drops ``no_tumor``),
applies the same ``crop_brain_contour`` + 240x240 resize as Figshare, computes 64-bit DCT pHash
against all 3,064 Figshare cropped slices, drops matches with Hamming distance <= threshold
(default 5), and saves:
  - ``data/external/brisc/{class_name}/*.png`` (kept images)
  - ``data/external/brisc_manifest.csv`` (file, patient_id=NA, label, class_name, min_hamming, nearest_figshare)
  - ``data/external/brisc_dedupe_report.json`` (counts before/after per class, dropped pairs)
  - ``data/external/brisc_test.npz`` (ready for fast evaluation)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, CROPPED_DIR, DATA_DIR, IMAGE_SIZE
from src.brain_tumor.crop import crop_brain_contour
from src.brain_tumor.phash import min_hamming_to_set, phash

BRISC_MAP = {
    "glioma": ("glioma", 0),
    "meningioma": ("meningioma", 1),
    "pituitary": ("pituitary tumor", 2),
}


def compute_figshare_hashes(cropped_dir: Path = CROPPED_DIR) -> tuple[np.ndarray, list[str]]:
    bits, names = [], []
    for cname in CLASS_NAMES:
        for p in sorted((cropped_dir / cname).glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            bits.append(phash(im))
            names.append(f"{cname}/{p.name}")
    return np.asarray(bits, dtype=bool), names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brisc-test", type=Path, default=ROOT / "brisc2025" / "classification_task" / "test")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "external")
    ap.add_argument("--hamming-thresh", type=int, default=5)
    args = ap.parse_args()

    out_img_root = args.out_dir / "brisc"
    if out_img_root.exists():
        shutil.rmtree(out_img_root)
    for cname in CLASS_NAMES:
        (out_img_root / cname).mkdir(parents=True, exist_ok=True)

    print("Hashing 3,064 Figshare slices...")
    fig_bits, fig_names = compute_figshare_hashes()
    print(f"Figshare reference hashes: {len(fig_names)}")

    candidates = []
    q_bits = []
    for brisc_folder, (cname, label) in BRISC_MAP.items():
        src_folder = args.brisc_test / brisc_folder
        files = sorted(p for p in src_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for p in tqdm(files, desc=f"BRISC {brisc_folder}"):
            im = cv2.imread(str(p))
            if im is None:
                continue
            cropped = crop_brain_contour(im)
            resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            q_bits.append(phash(gray))
            candidates.append(
                {
                    "src_path": str(p),
                    "filename": p.stem + ".png",
                    "class_name": cname,
                    "label": label,
                    "gray": gray,
                }
            )

    q_bits_arr = np.asarray(q_bits, dtype=bool)
    min_dist, argmin_idx = min_hamming_to_set(q_bits_arr, fig_bits)

    kept_rows = []
    kept_imgs, kept_y, kept_files = [], [], []
    dropped_details = []
    stats = {
        c: {"total": 0, "dropped_near_dup": 0, "kept": 0}
        for c in CLASS_NAMES
    }

    for cand, dist, ridx in zip(candidates, min_dist.tolist(), argmin_idx.tolist()):
        cname = cand["class_name"]
        stats[cname]["total"] += 1
        nearest = fig_names[ridx]
        if dist <= args.hamming_thresh:
            stats[cname]["dropped_near_dup"] += 1
            dropped_details.append(
                {
                    "brisc_file": cand["src_path"],
                    "class_name": cname,
                    "hamming": int(dist),
                    "nearest_figshare": nearest,
                }
            )
            continue
        stats[cname]["kept"] += 1
        dst = out_img_root / cname / cand["filename"]
        cv2.imwrite(str(dst), cand["gray"])
        kept_rows.append(
            {
                "file": str(dst),
                "patient_id": "NA",
                "label": cand["label"],
                "class_name": cname,
                "min_hamming_to_figshare": int(dist),
                "nearest_figshare": nearest,
            }
        )
        kept_imgs.append(cand["gray"])
        kept_y.append(cand["label"])
        kept_files.append(cand["filename"])

    manifest_path = args.out_dir / "brisc_manifest.csv"
    with open(manifest_path, "w", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["file", "patient_id", "label", "class_name", "min_hamming_to_figshare", "nearest_figshare"],
        )
        w.writeheader()
        w.writerows(kept_rows)

    npz_path = args.out_dir / "brisc_test.npz"
    np.savez_compressed(
        npz_path,
        x_test=np.asarray(kept_imgs, dtype=np.uint8),
        y_test=np.asarray(kept_y, dtype=np.int8),
        pid_test=np.asarray(["NA"] * len(kept_y)),
        file_test=np.asarray(kept_files),
        class_names=np.asarray(CLASS_NAMES),
    )

    report = {
        "note": "Image-level independent only (BRISC filenames do not carry patient IDs). 'no_tumor' excluded.",
        "hamming_threshold": args.hamming_thresh,
        "figshare_reference_slices": len(fig_names),
        "per_class": stats,
        "total_before": sum(v["total"] for v in stats.values()),
        "total_dropped": sum(v["dropped_near_dup"] for v in stats.values()),
        "total_kept": sum(v["kept"] for v in stats.values()),
        "min_hamming_histogram": {
            str(d): int((min_dist == d).sum()) for d in range(0, 16)
        },
        "dropped_examples": dropped_details[:50],
    }
    report_path = args.out_dir / "brisc_dedupe_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("per_class", "total_before", "total_dropped", "total_kept")}, indent=2))
    print(f"Wrote {npz_path} and {report_path}")


if __name__ == "__main__":
    main()
