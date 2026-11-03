#!/usr/bin/env python3
"""Check status of SageMaker M4 training jobs and sync S3 checkpoints/outputs to local runs/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGION = "us-east-1"


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sync", action="store_true", help="Sync S3 checkpoints into local runs/<run_name>/checkpoints/")
    ap.add_argument("--exclude-epoch-weights", action="store_true", help="Skip epoch_XX.weights.h5 when syncing locally to save bandwidth")
    args = ap.parse_args()

    meta_path = ROOT / "runs" / "sagemaker_jobs.json"
    if not meta_path.exists():
        print(f"No {meta_path} found.")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    bucket = meta["bucket"]
    prefix = meta["prefix"]

    for run_name, job_name in meta["jobs"].items():
        r = sh(["aws", "sagemaker", "describe-training-job", "--region", REGION, "--training-job-name", job_name], check=False)
        if r.returncode != 0:
            print(f"[{run_name}] {job_name}: error describing job: {r.stderr.strip()}")
            continue
        info = json.loads(r.stdout)
        status = info.get("TrainingJobStatus")
        sec = info.get("SecondaryStatus")
        bill = info.get("BillableTimeInSeconds", 0)
        fail = info.get("FailureReason", "")
        print(f"[{run_name:10s}] job={job_name}  status={status}/{sec}  billable={bill}s" + (f"  FAIL={fail}" if fail else ""))

        ckpt_s3 = f"s3://{bucket}/{prefix}/checkpoints/{run_name}/"
        ls_r = sh(["aws", "s3", "ls", ckpt_s3, "--region", REGION], check=False)
        if ls_r.returncode == 0 and ls_r.stdout.strip():
            lines = [ln.strip() for ln in ls_r.stdout.strip().splitlines()]
            epochs_saved = [ln.split()[-1] for ln in lines if "epoch_" in ln]
            print(f"  S3 checkpoints ({len(lines)} files, {len(epochs_saved)} epoch weights): {', '.join(epochs_saved[-5:])}")
        else:
            print("  S3 checkpoints: (none yet)")

        if args.sync:
            local_ckpt = ROOT / "runs" / run_name / "checkpoints"
            local_ckpt.mkdir(parents=True, exist_ok=True)
            cmd = ["aws", "s3", "sync", ckpt_s3, str(local_ckpt), "--region", REGION]
            if args.exclude_epoch_weights:
                cmd += ["--exclude", "epoch_*.weights.h5"]
            sh(cmd, check=False)
            state_f = local_ckpt / "train_state.json"
            if state_f.exists():
                st = json.loads(state_f.read_text())
                print(
                    f"  Synced train_state: next_epoch={st.get('next_epoch')} "
                    f"best_epoch={st.get('best_epoch')} stopped_early={st.get('stopped_early')} "
                    f"best={st.get('best')}"
                )
            # Copy summary artifacts to runs/<run_name>/ root if present
            local_root = ROOT / "runs" / run_name
            for fname in ("best.weights.h5", "history.csv", "train_state.json", "test_metrics.json", "test_predictions.csv", "config.json"):
                if (local_ckpt / fname).exists():
                    (local_root / fname).write_bytes((local_ckpt / fname).read_bytes())


if __name__ == "__main__":
    main()
