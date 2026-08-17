from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _check_no_overlap(splits: Mapping[str, Sequence[int]]) -> None:
    """Validate that no example index appears in more than one split.

    Args:
        splits: Mapping from split names to example index sequences.
    """

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
    """Create reproducible random train/validation/test split indices.

    Args:
        n: Total number of examples to split.
        val_frac: Fraction of examples assigned to validation.
        test_frac: Fraction of examples assigned to test.
        seed: Optional random seed for the permutation.
    """

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
    """Create split indices from a metadata column.

    Args:
        metadata: Sequence of per-example metadata mappings.
        split_col: Metadata key containing split labels.
        train_values: Labels interpreted as training examples.
        val_values: Labels interpreted as validation examples.
        test_values: Labels interpreted as test examples.
    """

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


def group_split_indices(
    metadata: Sequence[Mapping[str, object]],
    *,
    group_col: str = "group_gene_id",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Split complete biological groups while approximately balancing rows.

    This is the safe default for overlapping RBPNet windows. Groups are
    shuffled reproducibly and then assigned to test, validation, and training
    without ever dividing a group between splits.
    """

    if not metadata:
        raise ValueError("metadata must contain at least one example")
    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1):
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        value = row.get(group_col)
        if value is None or str(value).strip() == "":
            raise ValueError(
                f"Missing group column '{group_col}' for example index {index}"
            )
        groups.setdefault(str(value), []).append(index)
    if len(groups) < 1 + int(val_frac > 0) + int(test_frac > 0):
        raise ValueError("not enough biological groups for requested train/val/test splits")

    keys = np.asarray(sorted(groups), dtype=object)
    np.random.default_rng(seed).shuffle(keys)
    target_test = int(round(test_frac * len(metadata)))
    target_val = int(round(val_frac * len(metadata)))
    if test_frac > 0:
        target_test = max(1, target_test)
    if val_frac > 0:
        target_val = max(1, target_val)
    splits = {"train": [], "val": [], "test": []}
    for key in keys.tolist():
        rows = groups[str(key)]
        if test_frac > 0 and len(splits["test"]) < target_test:
            destination = "test"
        elif val_frac > 0 and len(splits["val"]) < target_val:
            destination = "val"
        else:
            destination = "train"
        splits[destination].extend(rows)
    if not splits["train"]:
        raise ValueError("group split leaves no training examples")
    _check_no_overlap(splits)
    validate_group_disjoint(splits, metadata, group_col=group_col)
    return splits


def validate_group_disjoint(
    splits: Mapping[str, Sequence[int]],
    metadata: Sequence[Mapping[str, object]],
    *,
    group_col: str = "group_gene_id",
) -> None:
    """Raise when one biological group occurs in multiple dataset splits."""

    owner: dict[str, str] = {}
    for split_name, indices in splits.items():
        for raw_index in indices:
            index = int(raw_index)
            if index < 0 or index >= len(metadata):
                raise ValueError(f"split index {index} is outside metadata bounds")
            value = metadata[index].get(group_col)
            if value is None or str(value).strip() == "":
                raise ValueError(
                    f"Missing group column '{group_col}' for example index {index}"
                )
            group = str(value)
            previous = owner.setdefault(group, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Biological group {group!r} appears in both '{previous}' and "
                    f"'{split_name}' using metadata column '{group_col}'"
                )


def normalize_splits(splits: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    """Normalize split indices to mutable integer lists with standard keys.

    Args:
        splits: Mapping from split names to index sequences.
    """

    out = {name: [int(i) for i in values] for name, values in splits.items()}
    out.setdefault("train", [])
    out.setdefault("val", [])
    out.setdefault("test", [])
    _check_no_overlap(out)
    return out
