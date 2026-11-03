#!/usr/bin/env python3
"""M4 training: EfficientNetB2, CE vs class-balanced focal loss, patient-grouped validation.

Runs locally or as a SageMaker training job (script mode, TF 2.16 DLC). Container-safe:
needs only numpy + TensorFlow/tf.keras. Every epoch it writes to ``--ckpt-dir``
(``/opt/ml/checkpoints`` on SageMaker, continuously synced to S3):

  last.weights.h5       latest weights (used for auto-resume across restarts)
  epoch_XX.weights.h5   weights after every epoch (nothing is ever lost)
  best.weights.h5       best patient-val macro-F1 (tie-break meningioma recall)
  train_state.json      epoch counter, best metric, plateau / early-stop counters, LR
  history.csv           per-epoch train loss/acc + patient-val metrics

If the job restarts (spot interruption, crash, manual re-launch) it resumes from
``train_state.json`` + ``last.weights.h5``. The test set is touched only once, after training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ON_SAGEMAKER = "SM_TRAINING_ENV" in os.environ or Path("/opt/ml/input/config").exists()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--loss", choices=("ce", "focal"), default="ce")
    a("--gamma", type=float, default=2.0)
    a("--beta", type=float, default=0.999)
    a("--run-name", "--run_name", default=None)
    a("--backbone", default="B2")
    a("--weights", default="imagenet", help="'imagenet' or 'none' (for offline smoke test)")
    a("--epochs", type=int, default=25)
    a("--batch-size", "--batch_size", type=int, default=32)
    a("--lr", type=float, default=1e-3)
    a("--lr-factor", "--lr_factor", type=float, default=0.3)
    a("--lr-patience", "--lr_patience", type=int, default=3)
    a("--es-patience", "--es_patience", type=int, default=7)
    a("--min-delta", "--min_delta", type=float, default=1e-3)
    a("--min-lr", "--min_lr", type=float, default=1e-6)
    a("--seed", type=int, default=42)
    a("--deterministic", choices=("auto", "on", "off"), default="auto")
    a("--subset", type=float, default=1.0, help="fraction of train/val/test for smoke tests")
    a("--keep-epoch-weights", "--keep_epoch_weights", type=int, default=1)
    a("--cpu", action="store_true", help="hide GPUs (local determinism / smoke tests)")
    a("--data", type=Path, default=None, help="path to m4_data.npz")
    a("--ckpt-dir", "--ckpt_dir", type=Path, default=None)
    a("--out-dir", "--out_dir", type=Path, default=None)
    a("--n-boot", "--n_boot", type=int, default=1000)
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[train_m4] ignoring unknown args:", unknown)
    if args.run_name is None:
        args.run_name = "A_ce" if args.loss == "ce" else f"B_cbfocal_g{args.gamma:g}"
    if args.data is None:
        channel = os.environ.get("SM_CHANNEL_TRAIN")
        args.data = Path(channel) / "m4_data.npz" if channel else ROOT / "data" / "m4" / "m4_data.npz"
    if args.ckpt_dir is None:
        args.ckpt_dir = Path("/opt/ml/checkpoints") if ON_SAGEMAKER else ROOT / "runs" / args.run_name / "checkpoints"
    if args.out_dir is None:
        args.out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")) if ON_SAGEMAKER else ROOT / "runs" / args.run_name
    return args


def load_npz(path: Path, subset: float, seed: int) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if subset < 1.0:
        rng = np.random.default_rng(seed)
        for split in ("train", "val", "test"):
            n = len(d[f"y_{split}"])
            keep = np.sort(rng.choice(n, max(8, int(n * subset)), replace=False))
            for prefix in ("x", "y", "pid", "file"):
                d[f"{prefix}_{split}"] = d[f"{prefix}_{split}"][keep]
    return d


def to_model_input(x_gray: np.ndarray) -> np.ndarray:
    """uint8 (N,H,W) -> float32 (N,H,W,3) in 0..255 (EfficientNet rescales internally)."""
    return np.repeat(x_gray[..., None], 3, axis=-1).astype(np.float32)


def make_tf_dataset(x_gray: np.ndarray, y_onehot: np.ndarray, order: np.ndarray, batch_size: int):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x_gray[order], y_onehot[order]))
    ds = ds.map(
        lambda img, y: (tf.cast(tf.repeat(img[..., None], 3, axis=-1), tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_probs(model, x_gray: np.ndarray, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(x_gray), batch_size):
        out.append(np.asarray(model.predict_on_batch(to_model_input(x_gray[i : i + batch_size]))))
    return np.concatenate(out, axis=0).astype(np.float64)


def set_optimizer_lr(optimizer, lr: float) -> None:
    lr_attr = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
    if hasattr(lr_attr, "assign"):
        lr_attr.assign(lr)
    else:
        import tensorflow as tf

        tf.keras.backend.set_value(lr_attr, lr)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_history(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def is_better(cur: dict, best: dict | None, min_delta: float) -> bool:
    if best is None:
        return True
    if cur["val_macro_f1"] > best["val_macro_f1"] + min_delta:
        return True
    if abs(cur["val_macro_f1"] - best["val_macro_f1"]) <= min_delta:
        return cur["val_meningioma_recall"] > best["val_meningioma_recall"] + 1e-9
    return False


def main() -> None:
    args = parse_args()
    t_start = time.time()

    import tensorflow as tf

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    from src.brain_tumor.losses import CategoricalFocalLoss, make_loss  # noqa: F401
    from src.brain_tumor.m4_metrics import CLASS_NAMES, bootstrap_ci, classification_summary
    from src.brain_tumor.model import build_efficientnet
    from src.brain_tumor.repro import enable_determinism, set_global_seed, sha256_file, write_run_config

    det_status = enable_determinism("off" if args.deterministic == "off" else args.deterministic)
    set_global_seed(args.seed)

    ckpt, out = args.ckpt_dir, args.out_dir
    ckpt.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.data, args.subset, args.seed)
    n_cls = len(CLASS_NAMES)
    x_tr, y_tr = data["x_train"], data["y_train"].astype(int)
    x_va, y_va = data["x_val"], data["y_val"].astype(int)
    x_te, y_te = data["x_test"], data["y_test"].astype(int)
    y_tr_oh = np.eye(n_cls, dtype=np.float32)[y_tr]
    train_counts = data["train_counts_preaug"].tolist()
    print(f"[train_m4] train {x_tr.shape} val {x_va.shape} test {x_te.shape}  preaug counts {train_counts}", flush=True)

    loss_obj, loss_info = make_loss(args.loss, train_counts=train_counts, gamma=args.gamma, beta=args.beta)

    state_path = ckpt / "train_state.json"
    last_weights = ckpt / "last.weights.h5"
    resumed = state_path.exists() and last_weights.exists()
    w_arg = None if (resumed or str(args.weights).lower() == "none") else args.weights
    model = build_efficientnet(args.backbone, weights=w_arg)

    if resumed:
        state = json.loads(state_path.read_text())
        print(f"[train_m4] RESUMING from epoch {state['next_epoch']} (best={state['best']})", flush=True)
        model.compile(
            loss=loss_obj,
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(state["lr"])),
            metrics=["accuracy"],
        )
        model.load_weights(str(last_weights))
        state["resumes"] = state.get("resumes", 0) + 1
    else:
        state = {
            "next_epoch": 0,
            "best": None,
            "best_epoch": None,
            "wait_lr": 0,
            "wait_es": 0,
            "lr": args.lr,
            "stopped_early": False,
            "history": [],
            "resumes": 0,
        }
        model.compile(
            loss=loss_obj,
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            metrics=["accuracy"],
        )

    config_extra = {
        "loss_info": loss_info,
        "determinism": det_status,
        "data_npz": str(args.data),
        "data_npz_sha256": sha256_file(args.data),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "val_patients": int(len(np.unique(data["pid_val"]))),
        "test_patients": int(len(np.unique(data["pid_test"]))),
        "train_counts_preaug": train_counts,
        "on_sagemaker": ON_SAGEMAKER,
        "instance": os.environ.get("SM_CURRENT_INSTANCE_TYPE"),
        "resumed": resumed,
        "checkpoint_rule": "max patient-val macro-F1 (min_delta), tie-break meningioma recall",
        "augmentation": "offline paper augmentation files only (rot 10, shift 0.1, h/v flip); no online aug",
    }
    write_run_config(ckpt, vars(args), config_extra)
    if ckpt.resolve() != out.resolve():
        shutil.copy2(ckpt / "config.json", out / "config.json")

    for epoch in range(state["next_epoch"], args.epochs):
        if state["stopped_early"]:
            break
        t0 = time.time()
        set_optimizer_lr(model.optimizer, float(state["lr"]))
        order = np.random.default_rng(args.seed * 1000 + epoch).permutation(len(y_tr))
        ds = make_tf_dataset(x_tr, y_tr_oh, order, args.batch_size)
        h = model.fit(ds, epochs=epoch + 1, initial_epoch=epoch, verbose=2)
        probs_va = predict_probs(model, x_va)
        s = classification_summary(y_va, probs_va)
        row = {
            "epoch": epoch + 1,
            "lr": float(state["lr"]),
            "loss": float(h.history["loss"][-1]),
            "accuracy": float(h.history["accuracy"][-1]),
            "val_accuracy": s["accuracy"],
            "val_macro_f1": s["macro_f1"],
            "val_macro_recall": s["macro_recall"],
            "val_meningioma_recall": s["meningioma_recall"],
            "val_macro_auc": s["macro_auc_ovr"],
            "seconds": round(time.time() - t0, 1),
        }
        improved = is_better(row, state["best"], args.min_delta)
        if improved:
            state["best"], state["best_epoch"] = row, epoch + 1
            state["wait_lr"] = state["wait_es"] = 0
            model.save_weights(str(ckpt / "best.weights.h5"))
        else:
            state["wait_lr"] += 1
            state["wait_es"] += 1
            if state["wait_lr"] >= args.lr_patience and state["lr"] > args.min_lr:
                state["lr"] = max(state["lr"] * args.lr_factor, args.min_lr)
                state["wait_lr"] = 0
                print(f"[train_m4] plateau -> lr {state['lr']:.2e}", flush=True)
            if state["wait_es"] >= args.es_patience:
                state["stopped_early"] = True
                print(f"[train_m4] early stopping after epoch {epoch + 1}", flush=True)
        row["improved"] = int(improved)
        state["history"].append(row)
        state["next_epoch"] = epoch + 1
        if args.keep_epoch_weights:
            model.save_weights(str(ckpt / f"epoch_{epoch + 1:02d}.weights.h5"))
        model.save_weights(str(last_weights))
        write_history(ckpt / "history.csv", state["history"])
        atomic_write_text(state_path, json.dumps(state, indent=2))
        print(
            f"[train_m4] epoch {epoch + 1}/{args.epochs} loss={row['loss']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} val_men_rec={row['val_meningioma_recall']:.4f} "
            f"{'*BEST*' if improved else ''} ({row['seconds']}s)",
            flush=True,
        )

    model.load_weights(str(ckpt / "best.weights.h5"))
    probs_te = predict_probs(model, x_te)
    summary = classification_summary(y_te, probs_te)
    summary["bootstrap_95ci"] = bootstrap_ci(y_te, probs_te, groups=data["pid_test"], n_boot=args.n_boot, seed=args.seed)
    summary["best_epoch"] = state["best_epoch"]
    summary["best_val"] = state["best"]
    summary["run_name"] = args.run_name
    summary["train_minutes"] = round((time.time() - t_start) / 60, 1)
    atomic_write_text(ckpt / "test_metrics.json", json.dumps(summary, indent=2))
    with open(ckpt / "test_predictions.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["file", "patient_id", "y_true", "y_pred"] + [f"p_{c}" for c in CLASS_NAMES])
        for f, p, yt, pr in zip(data["file_test"], data["pid_test"], y_te, probs_te):
            w.writerow([f, p, int(yt), int(pr.argmax())] + [f"{v:.6f}" for v in pr])

    if ckpt.resolve() != out.resolve():
        for name in ("best.weights.h5", "last.weights.h5", "history.csv", "train_state.json", "test_metrics.json", "test_predictions.csv", "config.json"):
            if (ckpt / name).exists():
                shutil.copy2(ckpt / name, out / name)
    print(
        f"[train_m4] DONE {args.run_name}: best epoch {state['best_epoch']} | test acc={summary['accuracy']:.4f} "
        f"macroF1={summary['macro_f1']:.4f} men_recall={summary['meningioma_recall']:.4f} "
        f"macroAUC={summary['macro_auc_ovr']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
