#!/usr/bin/env python3
"""Evaluate an M4 run or legacy checkpoint on Figshare patient-wise test or BRISC external test.

Computes:
  - accuracy, macro P/R/F1/specificity, macro & per-class one-vs-rest ROC-AUC
  - per-class precision, recall/sensitivity, F1, specificity, support
  - confusion matrix (CSV + PNG)
  - 1000-iteration bootstrap 95% CIs (patient-grouped on Figshare; image-level on BRISC)
  - per-slice probability predictions CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, DATA_DIR, OUTPUT_DIR
from src.brain_tumor.losses import CategoricalFocalLoss  # noqa: F401
from src.brain_tumor.m4_metrics import bootstrap_ci, classification_summary
from src.brain_tumor.metrics import save_confusion_matrix
from src.brain_tumor.model import build_efficientnet, compile_model


def load_model_or_weights(model_path: Path, backbone: str = "B2"):
    import keras

    model_path = Path(model_path)
    if model_path.is_dir():
        for cand in ("best.weights.h5", "checkpoints/best.weights.h5", "last.weights.h5", "checkpoints/last.weights.h5"):
            if (model_path / cand).exists():
                model_path = model_path / cand
                break
    if model_path.name.endswith(".weights.h5"):
        template_h5 = OUTPUT_DIR / "models" / f"efficientnet{backbone.lower()}.h5"
        if template_h5.exists():
            m = keras.models.load_model(str(template_h5))
        else:
            m = build_efficientnet(backbone, weights="imagenet")
        m.load_weights(str(model_path))
        # Keras 3.4 (SageMaker DLC) leaves EfficientNet's Normalization layer at (mean=0, var=1)
        # when loading legacy imagenet H5 weights, whereas Keras 3.15 calls finalize_state() on load_weights.
        norm = m.get_layer("normalization")
        norm.adapt_mean.assign([0.0, 0.0, 0.0])
        norm.adapt_variance.assign([1.0, 1.0, 1.0])
        norm.finalize_state()
        return m, model_path
    return keras.models.load_model(str(model_path)), model_path


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if name == "figshare_test":
        npz = np.load(DATA_DIR / "m4" / "m4_data.npz", allow_pickle=False)
        return npz["x_test"], npz["y_test"].astype(int), npz["pid_test"], npz["file_test"], "patient-grouped"
    if name == "figshare_val":
        npz = np.load(DATA_DIR / "m4" / "m4_data.npz", allow_pickle=False)
        return npz["x_val"], npz["y_val"].astype(int), npz["pid_val"], npz["file_val"], "patient-grouped"
    if name == "brisc":
        npz = np.load(DATA_DIR / "external" / "brisc_test.npz", allow_pickle=False)
        n = len(npz["y_test"])
        pids = np.array([f"img_{i:04d}" for i in range(n)])  # image-level resample
        return npz["x_test"], npz["y_test"].astype(int), pids, npz["file_test"], "image-level independent only (no patient IDs in BRISC; pHash deduplicated vs Figshare)"
    raise ValueError(f"Unknown dataset {name}")


def predict_probs(model, x_gray: np.ndarray, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(x_gray), batch_size):
        batch = np.repeat(x_gray[i : i + batch_size, ..., None], 3, axis=-1).astype(np.float32)
        out.append(model.predict_on_batch(batch))
    return np.concatenate(out, axis=0).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Run dir, .weights.h5, .keras, or .h5 path")
    ap.add_argument("--dataset", choices=("figshare_test", "figshare_val", "brisc"), default="figshare_test")
    ap.add_argument("--tag", required=True, help="Output tag, e.g. A_ce, B_cbfocal, frozen_m2")
    ap.add_argument("--backbone", default="B2")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    x, y, pids, files, independence_note = load_dataset(args.dataset)
    model, resolved_path = load_model_or_weights(args.model, args.backbone)
    probs = predict_probs(model, x)

    summary = classification_summary(y, probs)
    groups = pids if args.dataset.startswith("figshare") else None
    summary["bootstrap_95ci"] = bootstrap_ci(y, probs, groups=groups, n_boot=args.n_boot, seed=args.seed)
    summary["tag"] = args.tag
    summary["dataset"] = args.dataset
    summary["independence"] = independence_note
    summary["model_path"] = str(resolved_path)

    out_dir = ROOT / "results" / "eval" / f"{args.tag}__{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    cm = np.asarray(summary["confusion_matrix"], dtype=int)
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["true \\ pred"] + CLASS_NAMES)
        for cname, row in zip(CLASS_NAMES, cm.tolist()):
            w.writerow([cname] + row)
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png", title=f"{args.tag} on {args.dataset}")

    with open(out_dir / "predictions.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["file", "patient_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASS_NAMES])
        for f, p, yt, pr in zip(files, pids, y, probs):
            w.writerow([f, p, int(yt), int(pr.argmax())] + [f"{v:.6f}" for v in pr])

    ci = summary["bootstrap_95ci"]
    print(
        f"[{args.tag} | {args.dataset}] ({independence_note})\n"
        f"  Acc      = {summary['accuracy']*100:6.2f}%  (95% CI {ci['accuracy']['lo']*100:5.2f}–{ci['accuracy']['hi']*100:5.2f}%)\n"
        f"  Macro-F1 = {summary['macro_f1']*100:6.2f}%  (95% CI {ci['macro_f1']['lo']*100:5.2f}–{ci['macro_f1']['hi']*100:5.2f}%)\n"
        f"  Macro-R  = {summary['macro_recall']*100:6.2f}%  (95% CI {ci['macro_recall']['lo']*100:5.2f}–{ci['macro_recall']['hi']*100:5.2f}%)\n"
        f"  Men-R    = {summary['meningioma_recall']*100:6.2f}%  (95% CI {ci['meningioma_recall']['lo']*100:5.2f}–{ci['meningioma_recall']['hi']*100:5.2f}%)\n"
        f"  MacroAUC = {summary['macro_auc_ovr']:6.4f}   (95% CI {ci['macro_auc_ovr']['lo']:6.4f}–{ci['macro_auc_ovr']['hi']:6.4f})"
    )


if __name__ == "__main__":
    main()
