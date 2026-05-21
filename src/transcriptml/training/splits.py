from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _check_no_overlap(splits: Mapping[str, Sequence[int]]) -> None:
    seen: dict[int, str] = {}
    for name, values in splits.items():
        for idx in values:
            i = int(idx)
            if i in seen:
                raise ValueError(f"Index {i} appears in both '{seen[i]}' and '{name}'")
            seen[i] = name


def random_split_indices(
    n: int,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int | None = None,
) -> dict[str, list[int]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1):
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    if test_frac > 0:
        n_test = max(1, n_test)
    if val_frac > 0:
        n_val = max(1, n_val)
    if n_test + n_val >= n:
        raise ValueError("Split leaves no training examples")
    splits = {
        "test": idx[:n_test].astype(int).tolist(),
        "val": idx[n_test : n_test + n_val].astype(int).tolist(),
        "train": idx[n_test + n_val :].astype(int).tolist(),
    }
    _check_no_overlap(splits)
    return splits


def predefined_split_indices(
    metadata: Sequence[Mapping[str, object]],
    *,
    split_col: str = "split",
    train_values: Sequence[str] = ("train",),
    val_values: Sequence[str] = ("val", "valid", "validation"),
    test_values: Sequence[str] = ("test",),
) -> dict[str, list[int]]:
    train_set = {x.lower() for x in train_values}
    val_set = {x.lower() for x in val_values}
    test_set = {x.lower() for x in test_values}
    splits = {"train": [], "val": [], "test": []}
    for i, row in enumerate(metadata):
        value = str(row.get(split_col, "")).lower()
        if value in train_set:
            splits["train"].append(i)
        elif value in val_set:
            splits["val"].append(i)
        elif value in test_set:
            splits["test"].append(i)
    if not splits["train"]:
        raise ValueError(f"No training examples found using metadata column '{split_col}'")
    _check_no_overlap(splits)
    return splits


def normalize_splits(splits: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    out = {name: [int(i) for i in values] for name, values in splits.items()}
    out.setdefault("train", [])
    out.setdefault("val", [])
    out.setdefault("test", [])
    _check_no_overlap(out)
    return out
