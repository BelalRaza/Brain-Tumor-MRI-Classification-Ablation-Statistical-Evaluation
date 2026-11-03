#!/usr/bin/env python3
"""Package code + dataset to S3 and launch Run A (ml.g5.xlarge) and Run B (ml.g5.2xlarge) in parallel.

Uses AWS CLI (already configured on host) so no local boto3/sagemaker SDK install is required.
Every epoch inside the container writes to /opt/ml/checkpoints, which SageMaker continuously
syncs to s3://<bucket>/brain-tumor-m4/checkpoints/<run_name>/ so no epoch is ever lost and
re-launching with the same --run-name automatically resumes from the latest epoch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACCOUNT = "948127566077"
REGION = "us-east-1"
BUCKET = f"sagemaker-{REGION}-{ACCOUNT}"
PREFIX = "brain-tumor-m4"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20260926T030971"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker"


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ensure_bucket() -> None:
    r = sh(["aws", "s3api", "head-bucket", "--bucket", BUCKET, "--region", REGION], check=False)
    if r.returncode != 0:
        print(f"Creating bucket s3://{BUCKET} ...")
        sh(["aws", "s3api", "create-bucket", "--bucket", BUCKET, "--region", REGION])


def build_source_tarball(out_tar: Path) -> Path:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        tf.add(ROOT / "scripts" / "train_m4.py", arcname="train_m4.py")
        tf.add(ROOT / "scripts" / "train_m4.py", arcname="scripts/train_m4.py")
        for py in sorted((ROOT / "src").rglob("*.py")):
            tf.add(py, arcname=str(py.relative_to(ROOT)))
    return out_tar


def upload_inputs(src_tar: Path, npz_path: Path, skip_data: bool = False) -> tuple[str, str]:
    src_s3 = f"s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz"
    data_s3 = f"s3://{BUCKET}/{PREFIX}/data/m4_data.npz"
    print(f"Uploading {src_tar.name} -> {src_s3}")
    sh(["aws", "s3", "cp", str(src_tar), src_s3, "--region", REGION])
    if not skip_data:
        head = sh(["aws", "s3", "ls", data_s3, "--region", REGION], check=False)
        if head.returncode == 0 and str(npz_path.stat().st_size) in head.stdout:
            print(f"Data already in S3 ({data_s3}, matching size) — skipping upload.")
        else:
            print(f"Uploading {npz_path.name} ({npz_path.stat().st_size / 1e6:.1f} MB) -> {data_s3}")
            sh(["aws", "s3", "cp", str(npz_path), data_s3, "--region", REGION])
    return src_s3, f"s3://{BUCKET}/{PREFIX}/data/"


def launch_job(
    job_name: str,
    run_name: str,
    loss: str,
    instance_type: str,
    src_s3: str,
    data_s3_dir: str,
    epochs: int = 25,
    gamma: float = 2.0,
    seed: int = 42,
) -> str:
    ckpt_s3 = f"s3://{BUCKET}/{PREFIX}/checkpoints/{run_name}"
    out_s3 = f"s3://{BUCKET}/{PREFIX}/output/{run_name}"
    hyperparams = {
        "sagemaker_program": '"train_m4.py"',
        "sagemaker_submit_directory": f'"{src_s3}"',
        "loss": loss,
        "gamma": str(gamma),
        "run-name": run_name,
        "epochs": str(epochs),
        "batch-size": "32",
        "lr": "0.001",
        "lr-patience": "3",
        "es-patience": "7",
        "seed": str(seed),
        "deterministic": "off",
        "keep-epoch-weights": "1",
    }
    spec = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE_URI,
            "TrainingInputMode": "File",
        },
        "RoleArn": ROLE_ARN,
        "HyperParameters": hyperparams,
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": data_s3_dir,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": out_s3},
        "CheckpointConfig": {
            "S3Uri": ckpt_s3,
            "LocalPath": "/opt/ml/checkpoints",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
    }
    r = sh(
        ["aws", "sagemaker", "create-training-job", "--region", REGION, "--cli-input-json", json.dumps(spec)],
        check=True,
    )
    arn = json.loads(r.stdout)["TrainingJobArn"]
    print(f"Launched {job_name} ({instance_type}, loss={loss}) -> {arn}")
    print(f"  Checkpoints S3: {ckpt_s3}")
    print(f"  Output S3:      {out_s3}")
    return job_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=("both", "A", "B"), default="both")
    ap.add_argument("--skip-data-upload", action="store_true")
    args = ap.parse_args()

    ensure_bucket()
    tar_path = build_source_tarball(ROOT / "data" / "m4" / "sourcedir.tar.gz")
    npz_path = ROOT / "data" / "m4" / "m4_data.npz"
    src_s3, data_s3_dir = upload_inputs(tar_path, npz_path, skip_data=args.skip_data_upload)

    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    jobs = {}
    if args.only in ("both", "A"):
        jobs["A_ce"] = launch_job(
            job_name=f"bt-m4-A-ce-{stamp}",
            run_name="A_ce",
            loss="ce",
            instance_type="ml.g5.xlarge",
            src_s3=src_s3,
            data_s3_dir=data_s3_dir,
            epochs=args.epochs,
            seed=args.seed,
        )
    if args.only in ("both", "B"):
        jobs["B_cbfocal"] = launch_job(
            job_name=f"bt-m4-B-cbfocal-{stamp}",
            run_name="B_cbfocal",
            loss="focal",
            instance_type="ml.g5.2xlarge",
            src_s3=src_s3,
            data_s3_dir=data_s3_dir,
            epochs=args.epochs,
            gamma=2.0,
            seed=args.seed,
        )

    state_file = ROOT / "runs" / "sagemaker_jobs.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(
            {
                "launched_utc": datetime.now(timezone.utc).isoformat(),
                "bucket": BUCKET,
                "prefix": PREFIX,
                "jobs": jobs,
            },
            indent=2,
        )
    )
    print(f"Saved job metadata to {state_file}")


if __name__ == "__main__":
    main()
