#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _replace_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a fold-specific TranscriptML bundle and train config.")
    parser.add_argument("--dataset", required=True, help="Original TranscriptML dataset bundle")
    parser.add_argument("--base-config", required=True, help="Base TranscriptML train config JSON")
    parser.add_argument("--cv-root", required=True, help="Directory containing fold*/ outputs")
    parser.add_argument("--fold", type=int, required=True, help="Zero-based fold index")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-offset", type=int, default=1)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    cv_root = Path(args.cv_root)
    fold_dir = cv_root / f"fold{args.fold}"
    fold_dataset = fold_dir / "dataset"
    model_dir = fold_dir / "model"
    fold_dataset.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name in ("X.npy", "y.npy", "ids.txt", "schema.json", "metadata.json", "config.json"):
        src = dataset / name
        if src.exists():
            _replace_link(src, fold_dataset / name)

    n_examples = int(np.load(dataset / "X.npy", mmap_mode="r").shape[0])
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_examples, dtype=np.int64)
    rng.shuffle(indices)
    folds = [fold.astype(int).tolist() for fold in np.array_split(indices, args.n_folds)]

    test_fold = args.fold
    val_fold = (args.fold + args.val_offset) % args.n_folds
    train = []
    for i, fold_indices in enumerate(folds):
        if i not in {test_fold, val_fold}:
            train.extend(fold_indices)

    splits = {
        "train": train,
        "val": folds[val_fold],
        "test": folds[test_fold],
    }
    _write_json(fold_dataset / "splits.json", splits)

    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    config["dataset"] = str(fold_dataset)
    config["output_dir"] = str(model_dir)
    config["seed"] = int(config.get("seed", args.seed)) + int(args.fold)
    config.setdefault("mmap_mode", "r")
    config_path = fold_dir / "train_config.json"
    _write_json(config_path, config)

    print(config_path)


if __name__ == "__main__":
    main()
