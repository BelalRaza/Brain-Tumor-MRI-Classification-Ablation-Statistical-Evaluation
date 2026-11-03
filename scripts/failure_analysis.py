#!/usr/bin/env python3
"""M5 failure analysis + publication figures + ablation tables (Figshare patient-wise & BRISC).

Generates under ``results/``:
  - ``ablation_table.md`` and ``ablation_table.csv`` (with 95% CIs + predicted vs actual)
  - ``flip_analysis.json`` (slices where Run B fixed or broke Run A's prediction)
  - ``patient_error_concentration.csv`` (per-patient error counts on Figshare test)
  - ``figures/training_curves.png`` (Run A vs Run B loss, val macro-F1, val meningioma recall)
  - ``figures/confusion_comparison.png`` (2x2 grid: Run A & Run B on Figshare & BRISC)
  - ``figures/per_class_recall_ci.png`` (grouped bar chart with 95% bootstrap CIs)
  - ``figures/roc_curves.png`` (Run A vs Run B OvR ROC curves)
  - ``figures/failure_gallery_gradcam.png`` (top high-confidence errors + Grad-CAM overlays)
  - ``summary.md`` (narrative + tables ready to paste into the IEEE paper)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain_tumor.config import CLASS_NAMES, DATA_DIR
from src.brain_tumor.gradcam import gradcam_heatmap, overlay_gradcam
from src.brain_tumor.m4_metrics import bootstrap_ci, classification_summary

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def load_eval(tag: str, dataset: str) -> tuple[dict, list[dict]]:
    d = RESULTS_DIR / "eval" / f"{tag}__{dataset}"
    metrics = json.loads((d / "metrics.json").read_text())
    with open(d / "predictions.csv") as handle:
        preds = list(csv.DictReader(handle))
    return metrics, preds


def per_class_recall_ci(preds: list[dict], grouped: bool, n_boot: int = 1000, seed: int = 42) -> dict:
    y = np.array([int(r["y_true"]) for r in preds])
    p = np.array([[float(r[f"p_{c}"]) for c in CLASS_NAMES] for r in preds])
    groups = np.array([r["patient_id"] for r in preds]) if grouped else np.arange(len(preds))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples = {c: [] for c in CLASS_NAMES}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        s = classification_summary(y[idx], p[idx])
        for c in CLASS_NAMES:
            samples[c].append(s["per_class"][c]["recall"])
    out = {}
    for c in CLASS_NAMES:
        arr = np.asarray(samples[c])
        lo, hi = np.quantile(arr, [0.025, 0.975])
        out[c] = {"lo": float(lo), "hi": float(hi)}
    return out


def fmt_ci(val: float, ci: dict, pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    fmt = ".2f" if pct else ".4f"
    suf = "%" if pct else ""
    return f"{val*scale:{fmt}}{suf} [{ci['lo']*scale:{fmt}}, {ci['hi']*scale:{fmt}}]"


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for run_name, label, color in (("A_ce", "Run A (CE)", "#1f77b4"), ("B_cbfocal", "Run B (CB-Focal γ=2)", "#d62728")):
        hpath = ROOT / "runs" / run_name / "history.csv"
        if not hpath.exists():
            continue
        with open(hpath) as handle:
            rows = list(csv.DictReader(handle))
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["loss"]) for r in rows], marker="o", ms=3.5, label=label, color=color)
        axes[1].plot(ep, [float(r["val_macro_f1"]) * 100 for r in rows], marker="o", ms=3.5, label=label, color=color)
        axes[2].plot(ep, [float(r["val_meningioma_recall"]) * 100 for r in rows], marker="o", ms=3.5, label=label, color=color)
    axes[0].set_title("Training Loss (different scales)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Patient-Grouped Val Macro-F1 (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[2].set_title("Patient-Grouped Val Meningioma Recall (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recall (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion_grid(evals: dict) -> None:
    short = ["Glioma", "Meningioma", "Pituitary"]
    panels = [
        ("A_ce", "figshare_test", "Run A (CE) — Figshare Test"),
        ("B_cbfocal", "figshare_test", "Run B (CB-Focal) — Figshare Test"),
        ("A_ce", "brisc", "Run A (CE) — BRISC External"),
        ("B_cbfocal", "brisc", "Run B (CB-Focal) — BRISC External"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (tag, ds, title) in zip(axes.ravel(), panels):
        cm = np.asarray(evals[(tag, ds)][0]["confusion_matrix"], dtype=int)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = cm.max() / 2.0
        for i in range(3):
            row_sum = cm[i].sum()
            for j in range(3):
                pct = 100.0 * cm[i, j] / row_sum if row_sum else 0.0
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_comparison.png", dpi=200)
    plt.close(fig)


def plot_per_class_recall(evals: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    short = ["Glioma", "Meningioma", "Pituitary"]
    x = np.arange(3)
    w = 0.26
    specs = [
        ("frozen_m2", "M2 Baseline (leaky val)", "#7f7f7f", -w),
        ("A_ce", "Run A: CE (patient val)", "#1f77b4", 0.0),
        ("B_cbfocal", "Run B: CB-Focal γ=2", "#d62728", +w),
    ]
    for ax, (ds, ds_title, grouped) in zip(
        axes,
        [("figshare_test", "Figshare Patient-Wise Test (n=560)", True), ("brisc", "BRISC External Test (n=702, dedup)", False)],
    ):
        for tag, label, color, offset in specs:
            if (tag, ds) not in evals:
                continue
            m, preds = evals[(tag, ds)]
            cis = per_class_recall_ci(preds, grouped=grouped, n_boot=500)
            vals = [m["per_class"][c]["recall"] * 100 for c in CLASS_NAMES]
            yerr_lo = [vals[i] - cis[c]["lo"] * 100 for i, c in enumerate(CLASS_NAMES)]
            yerr_hi = [cis[c]["hi"] * 100 - vals[i] for i, c in enumerate(CLASS_NAMES)]
            bars = ax.bar(x + offset, vals, width=w, label=label, color=color, alpha=0.9, yerr=[yerr_lo, yerr_hi], capsize=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, short)
        ax.set_ylim(50, 106)
        ax.set_ylabel("Recall / Sensitivity (%)")
        ax.set_title(ds_title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_recall_ci.png", dpi=200)
    plt.close(fig)


def plot_roc_curves(evals: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    short = ["Glioma", "Meningioma", "Pituitary"]
    for ax, (ds, title) in zip(axes, [("figshare_test", "Figshare Patient-Wise Test"), ("brisc", "BRISC External Test (dedup)")]):
        for tag, ls, tag_lbl in [("A_ce", "--", "Run A (CE)"), ("B_cbfocal", "-", "Run B (CB-Focal)")]:
            m, preds = evals[(tag, ds)]
            y = np.array([int(r["y_true"]) for r in preds])
            for i, (cname, sname, col) in enumerate(zip(CLASS_NAMES, short, colors)):
                scores = np.array([float(r[f"p_{cname}"]) for r in preds])
                fpr, tpr, _ = roc_curve(y == i, scores)
                auc_val = m["per_class"][cname]["auc_ovr"]
                ax.plot(fpr, tpr, ls=ls, color=col, lw=1.6, label=f"{tag_lbl} · {sname} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)


def compute_flips_and_patient_errors(evals: dict) -> tuple[dict, list[dict]]:
    _, preds_a = evals[("A_ce", "figshare_test")]
    _, preds_b = evals[("B_cbfocal", "figshare_test")]
    fixed_by_b, broken_by_b, both_wrong = [], [], []
    for ra, rb in zip(preds_a, preds_b):
        yt = int(ra["y_true"])
        pa, pb = int(ra["y_pred"]), int(rb["y_pred"])
        item = {
            "file": ra["file"],
            "patient_id": ra["patient_id"],
            "true": CLASS_NAMES[yt],
            "pred_A": CLASS_NAMES[pa],
            "pred_B": CLASS_NAMES[pb],
            "conf_A": round(float(ra[f"p_{CLASS_NAMES[pa]}"]), 4),
            "conf_B": round(float(rb[f"p_{CLASS_NAMES[pb]}"]), 4),
        }
        if pa != yt and pb == yt:
            fixed_by_b.append(item)
        elif pa == yt and pb != yt:
            broken_by_b.append(item)
        elif pa != yt and pb != yt:
            both_wrong.append(item)

    flips = {
        "n_fixed_by_B": len(fixed_by_b),
        "n_broken_by_B": len(broken_by_b),
        "n_both_wrong": len(both_wrong),
        "net_gain_slices": len(fixed_by_b) - len(broken_by_b),
        "fixed_by_B_by_true_class": dict(Counter(x["true"] for x in fixed_by_b)),
        "broken_by_B_by_true_class": dict(Counter(x["true"] for x in broken_by_b)),
        "fixed_examples": fixed_by_b[:25],
        "broken_examples": broken_by_b[:25],
        "both_wrong_examples": both_wrong[:25],
    }
    (RESULTS_DIR / "flip_analysis.json").write_text(json.dumps(flips, indent=2))

    # Patient-level error concentration on Run B
    pat_stats = defaultdict(lambda: {"class": "", "total": 0, "errors_A": 0, "errors_B": 0})
    for ra, rb in zip(preds_a, preds_b):
        pid = ra["patient_id"]
        yt = int(ra["y_true"])
        pat_stats[pid]["class"] = CLASS_NAMES[yt]
        pat_stats[pid]["total"] += 1
        pat_stats[pid]["errors_A"] += int(int(ra["y_pred"]) != yt)
        pat_stats[pid]["errors_B"] += int(int(rb["y_pred"]) != yt)
    pat_rows = [
        {
            "patient_id": pid,
            "class": d["class"],
            "total_slices": d["total"],
            "errors_A": d["errors_A"],
            "errors_B": d["errors_B"],
            "error_rate_B": round(d["errors_B"] / d["total"], 4),
        }
        for pid, d in pat_stats.items()
    ]
    pat_rows.sort(key=lambda r: (-r["errors_B"], -r["errors_A"]))
    with open(RESULTS_DIR / "patient_error_concentration.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["patient_id", "class", "total_slices", "errors_A", "errors_B", "error_rate_B"])
        w.writeheader()
        w.writerows(pat_rows)
    return flips, pat_rows


def plot_failure_gradcam_gallery(evals: dict) -> None:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from scripts.eval_m4 import load_model_or_weights

    model_b, _ = load_model_or_weights(ROOT / "runs" / "B_cbfocal")
    npz_fig = np.load(DATA_DIR / "m4" / "m4_data.npz", allow_pickle=False)
    fig_map = {str(f): x for f, x in zip(npz_fig["file_test"], npz_fig["x_test"])}

    _, preds_a = evals[("A_ce", "figshare_test")]
    _, preds_b = evals[("B_cbfocal", "figshare_test")]

    # Pick 3 interesting failure cases of Run B and 3 cases where Run B fixed Run A's error
    wrongs, fixed = [], []
    for ra, rb in zip(preds_a, preds_b):
        yt = int(ra["y_true"])
        pa, pb = int(ra["y_pred"]), int(rb["y_pred"])
        conf_b = float(rb[f"p_{CLASS_NAMES[pb]}"])
        if pb != yt:
            wrongs.append((conf_b, ra, rb))
        elif pa != yt and pb == yt:
            fixed.append((conf_b, ra, rb))
    wrongs.sort(key=lambda t: -t[0])
    fixed.sort(key=lambda t: -t[0])
    selected = [("Misclassified", t) for t in wrongs[:3]] + [("Fixed by CB-Focal", t) for t in fixed[:3]]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(9.5, 2.7 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    short = ["Glioma", "Meningioma", "Pituitary"]
    for row_idx, (kind, (conf_b, ra, rb)) in enumerate(selected):
        fname = rb["file"]
        gray = fig_map[fname]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        inp = np.repeat(gray[None, ..., None], 3, axis=-1).astype(np.float32)
        yt, pa, pb = int(rb["y_true"]), int(ra["y_pred"]), int(rb["y_pred"])
        heat = gradcam_heatmap(inp, model_b, "top_activation", pred_index=pb)
        over = overlay_gradcam(bgr, heat)

        axes[row_idx, 0].imshow(gray, cmap="gray")
        axes[row_idx, 0].set_title(f"{kind}: {fname} (PID {rb['patient_id']})\nTrue: {short[yt]}", fontsize=8.5)
        axes[row_idx, 1].imshow(heat, cmap="jet")
        axes[row_idx, 1].set_title(f"Grad-CAM (`top_activation`)\nRun A pred: {short[pa]}", fontsize=8.5)
        axes[row_idx, 2].imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
        axes[row_idx, 2].set_title(f"Overlay — Run B pred: {short[pb]} ({conf_b*100:.1f}%)", fontsize=8.5)
        for c in range(3):
            axes[row_idx, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_gallery_gradcam.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    evals = {}
    for tag in ("frozen_m2", "A_ce", "B_cbfocal"):
        for ds in ("figshare_test", "brisc"):
            p = RESULTS_DIR / "eval" / f"{tag}__{ds}" / "metrics.json"
            if p.exists():
                evals[(tag, ds)] = load_eval(tag, ds)

    plot_training_curves()
    plot_confusion_grid(evals)
    plot_per_class_recall(evals)
    plot_roc_curves(evals)
    flips, pat_rows = compute_flips_and_patient_errors(evals)
    plot_failure_gradcam_gallery(evals)

    # Write ablation CSV + markdown
    csv_rows = []
    md_lines = [
        "# M4 Ablation & External Validation Results",
        "",
        "| Dataset | Run | Checkpoint Selection | Loss | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-Recall (95% CI) | Meningioma Recall (95% CI) | Glioma Recall | Pituitary Recall | Macro OvR AUC (95% CI) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "frozen_m2": ("M2 Frozen Baseline", "Leaky slice val (0.1)", "CE"),
        "A_ce": ("Run A (H1+H3 Baseline)", "Patient-grouped val", "CE"),
        "B_cbfocal": ("Run B (H2 Hypothesis)", "Patient-grouped val", "CB-Focal (γ=2)"),
    }
    for ds, ds_label in [("figshare_test", "Figshare Test (patient-wise, n=560, 48 pts)"), ("brisc", "BRISC External (dedup, n=702, image-level)")]:
        for tag in ("frozen_m2", "A_ce", "B_cbfocal"):
            if (tag, ds) not in evals:
                continue
            m, _ = evals[(tag, ds)]
            ci = m["bootstrap_95ci"]
            rname, ck_rule, l_name = labels[tag]
            md_lines.append(
                f"| {ds_label} | {rname} | {ck_rule} | {l_name} | "
                f"{fmt_ci(m['accuracy'], ci['accuracy'])} | "
                f"{fmt_ci(m['macro_f1'], ci['macro_f1'])} | "
                f"{fmt_ci(m['macro_recall'], ci['macro_recall'])} | "
                f"**{fmt_ci(m['meningioma_recall'], ci['meningioma_recall'])}** | "
                f"{m['per_class']['glioma']['recall']*100:.2f}% | "
                f"{m['per_class']['pituitary tumor']['recall']*100:.2f}% | "
                f"{fmt_ci(m['macro_auc_ovr'], ci['macro_auc_ovr'], pct=False)} |"
            )
            csv_rows.append(
                {
                    "dataset": ds,
                    "run": tag,
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "macro_recall": round(m["macro_recall"], 4),
                    "meningioma_recall": round(m["meningioma_recall"], 4),
                    "glioma_recall": round(m["per_class"]["glioma"]["recall"], 4),
                    "pituitary_recall": round(m["per_class"]["pituitary tumor"]["recall"], 4),
                    "macro_auc_ovr": round(m["macro_auc_ovr"], 4),
                }
            )

    (RESULTS_DIR / "ablation_table.md").write_text("\n".join(md_lines) + "\n")
    with open(RESULTS_DIR / "ablation_table.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("Wrote results/ablation_table.md, flip_analysis.json, patient_error_concentration.csv, and figures/*.png")


if __name__ == "__main__":
    main()
