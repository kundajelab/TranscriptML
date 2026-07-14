from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BUNDLE_FILES = ("X.npy", "y.npy", "ids.txt", "schema.json", "metadata.json", "config.json")


def _replace_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _validate_cv_args(*, fold: int, n_folds: int, val_offset: int) -> None:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if fold < 0 or fold >= n_folds:
        raise ValueError("fold must be in [0, n_folds)")
    if val_offset % n_folds == 0:
        raise ValueError("val_offset must choose a validation fold different from the test fold")


def cv_splits(n_examples: int, *, fold: int, n_folds: int = 10, seed: int = 42, val_offset: int = 1) -> dict[str, list[int]]:
    """Create deterministic train/validation/test indices for one CV fold."""

    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    _validate_cv_args(fold=fold, n_folds=n_folds, val_offset=val_offset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_examples, dtype=np.int64)
    rng.shuffle(indices)
    folds = [part.astype(int).tolist() for part in np.array_split(indices, n_folds)]

    test_fold = int(fold)
    val_fold = (test_fold + int(val_offset)) % int(n_folds)
    train: list[int] = []
    for i, fold_indices in enumerate(folds):
        if i not in {test_fold, val_fold}:
            train.extend(fold_indices)
    return {"train": train, "val": folds[val_fold], "test": folds[test_fold]}


def _with_required_model(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    model = config.get("model")
    if isinstance(model, Mapping):
        params = dict(model.get("params", {}) or {})
    else:
        params = {}
    config["model"] = {"name": model_name, "params": params}
    return config


def prepare_cv_fold(
    *,
    dataset: str | Path,
    base_config: str | Path,
    cv_root: str | Path,
    fold: int,
    model: str,
    n_folds: int = 10,
    seed: int = 42,
    val_offset: int = 1,
) -> Path:
    """Write one fold-specific dataset bundle and training config.

    Args:
        dataset: Original TranscriptML dataset bundle.
        base_config: Base JSON training config.
        cv_root: Directory containing fold outputs.
        fold: Zero-based test-fold index.
        model: Required registered model name to write into the fold config.
        n_folds: Number of CV folds.
        seed: Seed used to shuffle examples before splitting.
        val_offset: Validation fold offset relative to the test fold.
    """

    model = str(model).strip()
    if not model:
        raise ValueError("model is required")
    _validate_cv_args(fold=int(fold), n_folds=int(n_folds), val_offset=int(val_offset))
    dataset_path = Path(dataset)
    cv_root_path = Path(cv_root)
    fold_dir = cv_root_path / f"fold{int(fold)}"
    fold_dataset = fold_dir / "dataset"
    model_dir = fold_dir / "model"
    fold_dataset.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name in BUNDLE_FILES:
        src = dataset_path / name
        if src.exists():
            _replace_link(src, fold_dataset / name)

    n_examples = int(np.load(dataset_path / "X.npy", mmap_mode="r").shape[0])
    splits = cv_splits(
        n_examples,
        fold=int(fold),
        n_folds=int(n_folds),
        seed=int(seed),
        val_offset=int(val_offset),
    )
    _write_json(fold_dataset / "splits.json", splits)

    config = json.loads(Path(base_config).read_text(encoding="utf-8"))
    config["dataset"] = str(fold_dataset)
    config["output_dir"] = str(model_dir)
    config["seed"] = int(config.get("seed", seed)) + int(fold)
    config.setdefault("mmap_mode", "r")
    config = _with_required_model(config, model)
    config_path = fold_dir / "train_config.json"
    _write_json(config_path, config)
    return config_path
