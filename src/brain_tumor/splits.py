"""Patient-grouped manifests for M4 (H1).

Builds on the existing patient-wise 80:20 split in data/patient_split/ (seed 3) and carves a
patient-grouped validation set out of the *training patients only*. The augmented training
files in data/patient_augmented/train/ are re-used (no re-augmentation): each augmented file
``aug_<stem>_<i>_<rand>.png`` is mapped back to its source slice ``<stem>.png`` and therefore
to its patient, and files of validation patients are dropped from training.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import CLASS_NAMES, DATA_DIR, METADATA_PATH, PATIENT_AUGMENTED_DIR, PATIENT_SPLIT_DIR

MANIFEST_DIR = DATA_DIR / "manifests"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


class PatientLeakageError(AssertionError):
    pass


def source_stem(filename: str) -> str:
    """'aug_1841_0_6798.png' -> '1841'; '1841.png' -> '1841'."""
    stem = Path(filename).stem
    if stem.startswith("aug_"):
        return stem[4:].rsplit("_", 2)[0]
    return stem


def load_metadata(path: Path = METADATA_PATH) -> dict:
    meta = json.loads(Path(path).read_text())
    return {Path(k).stem: v for k, v in meta.items()}


def _list(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def carve_patient_val(train_rows: list[dict], frac: float = 0.15, seed: int = 42) -> set[str]:
    """Pick ~frac of the training patients (stratified by class) with StratifiedGroupKFold."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1.0 / frac))
    y = [r["label"] for r in train_rows]
    groups = [r["patient_id"] for r in train_rows]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(iter(sgkf.split(train_rows, y, groups)))
    return {train_rows[i]["patient_id"] for i in val_idx}


def assert_no_patient_overlap(splits: dict[str, list[dict]]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            owners[str(r["patient_id"])].add(split)
    bad = {p: s for p, s in owners.items() if len(s) > 1}
    if bad:
        raise PatientLeakageError(f"{len(bad)} patients occur in >1 split, e.g. {list(bad.items())[:5]}")


def build_manifests(
    patient_split_dir: Path = PATIENT_SPLIT_DIR,
    augmented_dir: Path = PATIENT_AUGMENTED_DIR,
    metadata_path: Path = METADATA_PATH,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    meta = load_metadata(metadata_path)

    def rows_for(folder: Path, split: str) -> list[dict]:
        rows = []
        for label, name in enumerate(CLASS_NAMES):
            for p in _list(folder / name):
                stem = source_stem(p.name)
                info = meta[stem]
                if info["class_name"] != name:
                    raise ValueError(f"label mismatch for {p}: folder {name} vs metadata {info['class_name']}")
                rows.append(
                    {
                        "file": str(p),
                        "source_stem": stem,
                        "patient_id": str(info["patient_id"]),
                        "label": label,
                        "class_name": name,
                        "split": split,
                    }
                )
        return rows

    orig_train = rows_for(patient_split_dir / "train", "train")
    test = rows_for(patient_split_dir / "test", "test")
    val_patients = carve_patient_val(orig_train, frac=val_frac, seed=seed)

    val = [dict(r, split="val") for r in orig_train if r["patient_id"] in val_patients]
    train_orig_kept = [r for r in orig_train if r["patient_id"] not in val_patients]
    aug_train = [r for r in rows_for(augmented_dir / "train", "train") if r["patient_id"] not in val_patients]

    splits = {"train": aug_train, "train_source": train_orig_kept, "val": val, "test": test}
    assert_no_patient_overlap({"train": aug_train + train_orig_kept, "val": val, "test": test})
    kept_stems = {r["source_stem"] for r in train_orig_kept}
    orphan = [r for r in aug_train if r["source_stem"] not in kept_stems]
    if orphan:
        raise PatientLeakageError(f"{len(orphan)} augmented files have no training source slice")
    return splits


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifests(splits: dict[str, list[dict]], out_dir: Path = MANIFEST_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    fields = ["file", "source_stem", "patient_id", "label", "class_name", "split"]
    for name, rows in splits.items():
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fields})
        hashes[name] = sha256_text(path)
        (out_dir / f"{name}.csv.sha256").write_text(hashes[name] + "\n")
    return hashes


def split_counts(splits: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rows in splits.items():
        imgs = Counter(r["class_name"] for r in rows)
        pats = {c: len({r["patient_id"] for r in rows if r["class_name"] == c}) for c in CLASS_NAMES}
        out[name] = {
            c: {"images": imgs.get(c, 0), "patients": pats[c]} for c in CLASS_NAMES
        }
    return out
