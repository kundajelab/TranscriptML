"""Memory-bounded combination of fold-level attribution tensors."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from transcriptml.progress import ProgressReporter


CombinationMethod = Literal["mean", "median", "conservative"]
AlignmentMode = Literal["intersection", "union", "mode", "exact"]
ConfidenceSubset = Literal["high", "low", "all"]
COMBINATION_METHODS: tuple[CombinationMethod, ...] = (
    "mean",
    "median",
    "conservative",
)
_FOLD_PATTERN = re.compile(r"fold(\d+)")
_DEEP_SHAP_PATTERN = re.compile(
    r"(?P<partition>high_confidence|low_confidence)_"
    r"(?:actual|hypothetical)_attributions\.npy"
)


def find_fold_attribution_paths(
    input_dir: str | Path,
    input_file: str | Path,
) -> list[Path]:
    """Find one relative attribution file in contiguous numeric fold directories."""

    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Attribution directory does not exist: {root}")
    relative = Path(input_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("input_file must be a relative path within each fold directory")

    paths_by_fold: dict[int, Path] = {}
    for fold_dir in root.iterdir():
        match = _FOLD_PATTERN.fullmatch(fold_dir.name)
        if match is None or not fold_dir.is_dir():
            continue
        path = fold_dir / relative
        if not path.is_file():
            continue
        fold_index = int(match.group(1))
        if fold_index in paths_by_fold:
            raise ValueError(
                f"Multiple directories represent fold{fold_index}: "
                f"{paths_by_fold[fold_index].parent} and {fold_dir}"
            )
        paths_by_fold[fold_index] = path

    if not paths_by_fold:
        raise ValueError(f"No fold*/{relative} files found in {root}")
    fold_indices = sorted(paths_by_fold)
    expected = list(range(fold_indices[-1] + 1))
    if fold_indices != expected:
        missing = sorted(set(expected) - set(fold_indices))
        raise ValueError(
            "Fold indices must be contiguous and start at 0; "
            f"missing fold(s): {missing}"
        )
    return [paths_by_fold[index] for index in fold_indices]


def _validate_arrays(
    paths: list[Path],
    *,
    allow_different_rows: bool,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    reference_shape = arrays[0].shape
    for path, array in zip(paths, arrays, strict=True):
        if array.ndim != 3 or array.shape[1] != 4:
            raise ValueError(
                f"Expected attribution arrays with shape (N, 4, L), but "
                f"{path} has shape {array.shape}"
            )
        if array.shape[1:] != reference_shape[1:] or (
            not allow_different_rows and array.shape != reference_shape
        ):
            raise ValueError(
                "All fold arrays must have matching attribution dimensions; "
                f"expected {reference_shape}, "
                f"but {path} has shape {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Attribution array must have a numeric dtype: {path}")
    return arrays, (int(reference_shape[1]), int(reference_shape[2]))


def _validate_deep_shap_alignment(
    paths: list[Path],
    arrays: list[np.ndarray],
    input_file: Path,
    alignment: AlignmentMode,
    mode_threshold: float,
) -> tuple[str, np.ndarray, list[str], list[np.ndarray], np.ndarray] | None:
    """Resolve common rows for a partitioned DeepSHAP tensor."""

    match = _DEEP_SHAP_PATTERN.fullmatch(input_file.name)
    if match is None:
        return None
    partition = match.group("partition")
    indices_name = f"{partition}_indices.npy"
    ids_name = f"{partition}_ids.txt"
    index_paths = [path.with_name(indices_name) for path in paths]
    id_paths = [path.with_name(ids_name) for path in paths]
    missing = [path for path in (*index_paths, *id_paths) if not path.is_file()]
    if missing:
        raise ValueError(
            "Partitioned DeepSHAP averaging requires mapping sidecars; missing: "
            + ", ".join(str(path) for path in missing)
        )

    fold_indices: list[np.ndarray] = []
    fold_ids: list[list[str]] = []
    for path, array, index_path, id_path in zip(
        paths, arrays, index_paths, id_paths, strict=True
    ):
        indices = np.load(index_path, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Expected a one-dimensional integer array: {index_path}")
        ids = id_path.read_text(encoding="utf-8").splitlines()
        if len(indices) != array.shape[0] or len(ids) != len(indices):
            raise ValueError(
                f"Attribution rows, indices, and IDs have different lengths in {path.parent}"
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"DeepSHAP indices must be unique within each fold: {index_path}")
        fold_indices.append(indices.astype(np.int64, copy=False))
        fold_ids.append(ids)

    if alignment == "exact":
        reference_indices = fold_indices[0]
        reference_ids = fold_ids[0]
        for indices, ids, index_path in zip(
            fold_indices[1:], fold_ids[1:], index_paths[1:], strict=True
        ):
            if not np.array_equal(indices, reference_indices) or ids != reference_ids:
                raise ValueError(
                    "DeepSHAP rows are not identically aligned across folds for "
                    f"{input_file.name}; mismatch at {index_path.parent}"
                )
        positions = [np.arange(len(reference_indices), dtype=np.int64) for _ in paths]
        support = np.full(len(reference_indices), len(paths), dtype=np.int64)
        return partition, reference_indices, reference_ids, positions, support

    selected_indices, support = _select_indices_by_fold_support(
        fold_indices,
        alignment=alignment,
        mode_threshold=mode_threshold,
    )
    positions: list[np.ndarray] = []
    selected_ids_by_fold: list[dict[int, str]] = []
    for indices, ids in zip(fold_indices, fold_ids, strict=True):
        position_by_index = {int(index): row for row, index in enumerate(indices)}
        positions.append(
            np.asarray(
                [position_by_index.get(int(index), -1) for index in selected_indices],
                dtype=np.int64,
            )
        )
        selected_ids_by_fold.append(
            {int(index): ids[row] for row, index in enumerate(indices)}
        )

    selected_ids: list[str] = []
    for index in selected_indices:
        observed = [
            fold_ids_by_index[int(index)]
            for fold_ids_by_index in selected_ids_by_fold
            if int(index) in fold_ids_by_index
        ]
        if any(identifier != observed[0] for identifier in observed[1:]):
            raise ValueError(f"Sequence ID mismatch for global index {int(index)}")
        selected_ids.append(observed[0])

    return partition, selected_indices, selected_ids, positions, support


def _select_indices_by_fold_support(
    fold_indices: list[np.ndarray],
    *,
    alignment: AlignmentMode,
    mode_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select canonical indices and report their desired-partition fold support."""

    counts: dict[int, int] = {}
    for indices in fold_indices:
        for index in indices:
            value = int(index)
            counts[value] = counts.get(value, 0) + 1
    fold_count = len(fold_indices)
    if alignment == "intersection":
        selected = sorted(index for index, count in counts.items() if count == fold_count)
    elif alignment == "union":
        selected = sorted(counts)
    elif alignment == "mode":
        selected = sorted(
            index
            for index, count in counts.items()
            if count / fold_count > float(mode_threshold)
        )
    else:  # ``exact`` is handled before this helper is called.
        raise ValueError(f"Unsupported fold alignment mode: {alignment}")
    selected_indices = np.asarray(selected, dtype=np.int64)
    support = np.asarray([counts[index] for index in selected], dtype=np.int64)
    return selected_indices, support


def _combine_chunk(
    arrays: list[np.ndarray],
    row_positions: list[np.ndarray] | None,
    start: int,
    stop: int,
    *,
    method: CombinationMethod,
    dtype: np.dtype,
) -> np.ndarray:
    def select_values(array: np.ndarray, fold: int) -> np.ndarray:
        rows = slice(start, stop) if row_positions is None else row_positions[fold][start:stop]
        return np.asarray(array[rows], dtype=dtype)

    if method == "mean":
        combined = np.zeros((stop - start, *arrays[0].shape[1:]), dtype=dtype)
        if row_positions is not None and any(
            np.any(positions[start:stop] < 0) for positions in row_positions
        ):
            counts = np.zeros(stop - start, dtype=np.int64)
            for fold, array in enumerate(arrays):
                positions = row_positions[fold][start:stop]
                present = positions >= 0
                if np.any(present):
                    combined[present] += np.asarray(array[positions[present]], dtype=dtype)
                    counts[present] += 1
            if np.any(counts == 0):
                raise RuntimeError("Selected DeepSHAP row has no contributing folds")
            combined /= counts[:, None, None]
            return combined
        for fold, array in enumerate(arrays):
            combined += select_values(array, fold)
        combined /= len(arrays)
        return combined

    stacked = np.stack(
        [select_values(array, fold) for fold, array in enumerate(arrays)],
        axis=0,
    )
    if method == "median":
        return np.median(stacked, axis=0).astype(dtype, copy=False)
    minimum_index = np.argmin(np.abs(stacked), axis=0, keepdims=True)
    return np.take_along_axis(stacked, minimum_index, axis=0)[0]


def _close_memmap(array: np.ndarray | None) -> None:
    """Flush and close a NumPy memory map when one is present."""

    if array is None:
        return
    if isinstance(array, np.memmap):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None and not mmap.closed:
            array.flush()
            mmap.close()


class _FoldPartitionUnion:
    """Memory-mapped high/low DeepSHAP rows in canonical global-index order."""

    def __init__(
        self,
        high: np.ndarray,
        low: np.ndarray,
        high_indices: np.ndarray,
        low_indices: np.ndarray,
        high_ids: list[str],
        low_ids: list[str],
        *,
        fold_dir: Path,
    ) -> None:
        overlap = np.intersect1d(high_indices, low_indices)
        if overlap.size:
            raise ValueError(
                "High- and low-confidence DeepSHAP partitions overlap in "
                f"{fold_dir}; first duplicated global index: {int(overlap[0])}"
            )
        combined_indices = np.concatenate((high_indices, low_indices))
        order = np.argsort(combined_indices, kind="stable")
        self.indices = combined_indices[order]
        combined_ids = high_ids + low_ids
        self.ids = [combined_ids[int(row)] for row in order]
        high_count = len(high_indices)
        source_is_high = np.concatenate(
            (
                np.ones(high_count, dtype=np.bool_),
                np.zeros(len(low_indices), dtype=np.bool_),
            )
        )
        source_rows = np.concatenate(
            (
                np.arange(high_count, dtype=np.int64),
                np.arange(len(low_indices), dtype=np.int64),
            )
        )
        self.source_is_high = source_is_high[order]
        self.source_rows = source_rows[order]
        self.high = high
        self.low = low
        self.fold_dir = fold_dir

    def take(self, positions: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Read selected canonical rows without materializing the full union."""

        output = np.empty((len(positions), *self.high.shape[1:]), dtype=dtype)
        selected_high = self.source_is_high[positions]
        selected_rows = self.source_rows[positions]
        if np.any(selected_high):
            output[selected_high] = self.high[selected_rows[selected_high]]
        if np.any(~selected_high):
            output[~selected_high] = self.low[selected_rows[~selected_high]]
        return output


def _load_partition_mapping(
    array: np.ndarray,
    attribution_path: Path,
    partition: str,
) -> tuple[np.ndarray, list[str]]:
    index_path = attribution_path.with_name(f"{partition}_indices.npy")
    id_path = attribution_path.with_name(f"{partition}_ids.txt")
    missing = [path for path in (index_path, id_path) if not path.is_file()]
    if missing:
        raise ValueError(
            "Partitioned DeepSHAP averaging requires mapping sidecars; missing: "
            + ", ".join(str(path) for path in missing)
        )
    indices = np.load(index_path, allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Expected a one-dimensional integer array: {index_path}")
    ids = id_path.read_text(encoding="utf-8").splitlines()
    if len(indices) != array.shape[0] or len(ids) != len(indices):
        raise ValueError(
            "Attribution rows, indices, and IDs have different lengths in "
            f"{attribution_path.parent}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"DeepSHAP indices must be unique within each fold: {index_path}")
    return indices.astype(np.int64, copy=False), ids


def _load_deep_shap_unions(
    input_dir: str | Path,
    attribution_kind: Literal["actual", "hypothetical"],
) -> tuple[list[_FoldPartitionUnion], list[Path], tuple[int, int]]:
    """Load each fold's high/low partitions as a single indexed view."""

    high_name = f"high_confidence_{attribution_kind}_attributions.npy"
    low_name = f"low_confidence_{attribution_kind}_attributions.npy"
    high_paths = find_fold_attribution_paths(input_dir, high_name)
    low_paths = find_fold_attribution_paths(input_dir, low_name)
    if [path.parent for path in high_paths] != [path.parent for path in low_paths]:
        raise ValueError(
            "DeepSHAP confidence='all' requires high- and low-confidence files "
            "for the same contiguous fold directories"
        )
    all_paths = [path for pair in zip(high_paths, low_paths, strict=True) for path in pair]
    all_arrays, shape_tail = _validate_arrays(all_paths, allow_different_rows=True)
    unions: list[_FoldPartitionUnion] = []
    for fold, (high_path, low_path) in enumerate(
        zip(high_paths, low_paths, strict=True)
    ):
        high = all_arrays[2 * fold]
        low = all_arrays[2 * fold + 1]
        high_indices, high_ids = _load_partition_mapping(
            high, high_path, "high_confidence"
        )
        low_indices, low_ids = _load_partition_mapping(
            low, low_path, "low_confidence"
        )
        unions.append(
            _FoldPartitionUnion(
                high,
                low,
                high_indices,
                low_indices,
                high_ids,
                low_ids,
                fold_dir=high_path.parent,
            )
        )
    return unions, all_paths, shape_tail


def _align_partition_unions(
    unions: list[_FoldPartitionUnion],
    alignment: AlignmentMode,
    mode_threshold: float,
) -> tuple[np.ndarray, list[str], list[np.ndarray], np.ndarray]:
    """Align unioned eligible rows across folds using global sequence indices."""

    reference = unions[0]
    if alignment == "exact":
        for union in unions[1:]:
            if not np.array_equal(union.indices, reference.indices) or union.ids != reference.ids:
                raise ValueError(
                    "Combined high/low DeepSHAP rows are not identically aligned "
                    f"across folds; mismatch at {union.fold_dir}"
                )
        positions = [np.arange(len(reference.indices), dtype=np.int64) for _ in unions]
        support = np.full(len(reference.indices), len(unions), dtype=np.int64)
        return reference.indices, reference.ids, positions, support

    selected_indices, support = _select_indices_by_fold_support(
        [union.indices for union in unions],
        alignment=alignment,
        mode_threshold=mode_threshold,
    )
    positions: list[np.ndarray] = []
    ids_by_fold: list[dict[int, str]] = []
    for union in unions:
        position_by_index = {int(index): row for row, index in enumerate(union.indices)}
        positions.append(
            np.asarray(
                [position_by_index.get(int(index), -1) for index in selected_indices],
                dtype=np.int64,
            )
        )
        ids_by_fold.append(
            {int(index): union.ids[row] for row, index in enumerate(union.indices)}
        )
    selected_ids: list[str] = []
    for index in selected_indices:
        observed = [
            values[int(index)] for values in ids_by_fold if int(index) in values
        ]
        if any(identifier != observed[0] for identifier in observed[1:]):
            raise ValueError(f"Sequence ID mismatch for global index {int(index)}")
        selected_ids.append(observed[0])
    return selected_indices, selected_ids, positions, support


def average_attributions(
    input_dir: str | Path,
    *,
    input_file: str | Path | None = None,
    out_dir: str | Path | None = None,
    ism: bool = False,
    method: CombinationMethod = "mean",
    alignment: AlignmentMode = "intersection",
    mode_threshold: float = 0.5,
    confidence: ConfidenceSubset | None = None,
    use_all_folds: bool = False,
    dataset: str | Path | None = None,
    all_channels: bool = False,
    batch_size: int = 256,
    dtype: str | np.dtype = "float32",
    progress: bool = True,
) -> dict[str, Any]:
    """Combine aligned fold-level attribution tensors.

    DeepSHAP tensors are always averaged. With ``ism=True``, ``method`` may be
    mean, median, or conservative (the fold value closest to zero), and both
    raw and nucleotide-channel-centered tensors are written.
    """

    method_value = str(method).strip().lower()
    if method_value not in COMBINATION_METHODS:
        raise ValueError(f"method must be one of: {', '.join(COMBINATION_METHODS)}")
    selected_method = cast(CombinationMethod, method_value)
    if not ism and selected_method != "mean":
        raise ValueError("DeepSHAP attributions are averaged; --method requires --ism")
    if ism and confidence is not None:
        raise ValueError("confidence is only available for DeepSHAP averaging")
    if ism and use_all_folds:
        raise ValueError("use_all_folds is only available for DeepSHAP averaging")
    alignment_value = str(alignment).strip().lower()
    if alignment_value not in {"intersection", "union", "mode", "exact"}:
        raise ValueError(
            "alignment must be one of: intersection, union, mode, exact"
        )
    selected_alignment = cast(AlignmentMode, alignment_value)
    if not 0.0 <= float(mode_threshold) < 1.0:
        raise ValueError("mode_threshold must be in [0, 1)")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError(f"dtype must be floating-point, got {output_dtype}")

    if not ism and input_file is None:
        if confidence not in {"high", "low", "all"}:
            raise ValueError(
                "DeepSHAP averaging requires confidence='high', 'low', or 'all'"
            )
        if dataset is None:
            raise ValueError("DeepSHAP averaging requires a dataset for aligned sequences")
        return _average_deep_shap_partition(
            input_dir,
            out_dir=out_dir,
            confidence=confidence,
            use_all_folds=bool(use_all_folds),
            dataset=dataset,
            alignment=selected_alignment,
            mode_threshold=float(mode_threshold),
            all_channels=all_channels,
            batch_size=int(batch_size),
            dtype=output_dtype,
            progress=progress,
        )

    if input_file is None and not ism:
        raise ValueError("input_file is required for DeepSHAP attribution averaging")
    if use_all_folds:
        raise ValueError(
            "use_all_folds requires high-level DeepSHAP mode with --confidence"
        )
    relative = Path("deltas.npy" if input_file is None else input_file)
    paths = find_fold_attribution_paths(input_dir, relative)
    partitioned = not ism and _DEEP_SHAP_PATTERN.fullmatch(relative.name) is not None
    arrays, shape_tail = _validate_arrays(
        paths,
        allow_different_rows=partitioned,
    )
    output_root = Path(input_dir) if out_dir is None else Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if ism:
        combined_path = output_root / "combined_deltas.npy"
        centered_path: Path | None = output_root / "combined_centered_deltas.npy"
    else:
        combined_path = output_root / f"average_{relative.name}"
        centered_path = None
    alignment_result = (
        _validate_deep_shap_alignment(
            paths,
            arrays,
            relative,
            selected_alignment,
            float(mode_threshold),
        )
        if partitioned
        else None
    )
    row_positions = None if alignment_result is None else alignment_result[3]
    n_rows = arrays[0].shape[0] if alignment_result is None else len(alignment_result[1])
    shape = (int(n_rows), *shape_tail)

    temporary_combined = combined_path.with_name(f".{combined_path.name}.partial.npy")
    temporary_centered = (
        None
        if centered_path is None
        else centered_path.with_name(f".{centered_path.name}.partial.npy")
    )
    combined_output = np.lib.format.open_memmap(
        temporary_combined,
        mode="w+",
        dtype=output_dtype,
        shape=shape,
    )
    centered_output = (
        None
        if temporary_centered is None
        else np.lib.format.open_memmap(
            temporary_centered,
            mode="w+",
            dtype=output_dtype,
            shape=shape,
        )
    )
    reporter = ProgressReporter(
        "average-attributions",
        total=shape[0],
        unit="sequences",
        enabled=progress,
    )
    try:
        for start in range(0, shape[0], int(batch_size)):
            stop = min(start + int(batch_size), shape[0])
            combined = _combine_chunk(
                arrays,
                row_positions,
                start,
                stop,
                method=selected_method,
                dtype=output_dtype,
            )
            combined_output[start:stop] = combined
            if centered_output is not None:
                centered_output[start:stop] = combined - np.mean(
                    combined,
                    axis=1,
                    keepdims=True,
                    dtype=output_dtype,
                )
            reporter.update(advance=stop - start)
        _close_memmap(combined_output)
        _close_memmap(centered_output)
        os.replace(temporary_combined, combined_path)
        if temporary_centered is not None and centered_path is not None:
            os.replace(temporary_centered, centered_path)
    except BaseException:
        _close_memmap(combined_output)
        _close_memmap(centered_output)
        temporary_combined.unlink(missing_ok=True)
        if temporary_centered is not None:
            temporary_centered.unlink(missing_ok=True)
        raise
    reporter.close(extra=f"{len(paths)} folds; method={selected_method}")

    outputs = {"combined": str(combined_path)}
    if centered_path is not None:
        outputs["centered"] = str(centered_path)
    if alignment_result is not None:
        partition, indices, ids, _, fold_support = alignment_result
        indices_output = output_root / f"{partition}_indices.npy"
        ids_output = output_root / f"{partition}_ids.txt"
        support_output = output_root / f"{partition}_fold_support.npy"
        confidence_support_output = (
            output_root / f"{partition}_confidence_support.npy"
        )
        np.save(indices_output, indices)
        np.save(support_output, fold_support)
        np.save(confidence_support_output, fold_support)
        ids_output.write_text(
            "".join(f"{identifier}\n" for identifier in ids),
            encoding="utf-8",
        )
        outputs.update(
            indices=str(indices_output),
            ids=str(ids_output),
            fold_support=str(support_output),
            confidence_support=str(confidence_support_output),
        )
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "ism" if ism else "deep_shap",
        "input_dir": str(input_dir),
        "input_file": str(relative),
        "fold_count": len(paths),
        "fold_files": [str(path) for path in paths],
        "combination_method": selected_method,
        "alignment": selected_alignment if partitioned else "row_order",
        "mode_threshold": float(mode_threshold) if partitioned else None,
        "fold_row_counts": [int(array.shape[0]) for array in arrays],
        "aligned_row_count": int(shape[0]),
        "fold_support_min": (
            int(np.min(alignment_result[4]))
            if alignment_result is not None and len(alignment_result[4])
            else None
        ),
        "fold_support_max": (
            int(np.max(alignment_result[4]))
            if alignment_result is not None and len(alignment_result[4])
            else None
        ),
        "channel_centered": bool(ism),
        "shape": list(shape),
        "dtype": str(output_dtype),
        "batch_size": int(batch_size),
        "outputs": outputs,
    }
    summary_name = (
        "average_attributions_summary.json"
        if ism
        else f"average_{relative.stem}_summary.json"
    )
    (output_root / summary_name).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_sequence_slice(
    dataset: str | Path,
    output_root: Path,
    *,
    partition: str,
    all_channels: bool,
    batch_size: int,
) -> dict[str, str]:
    """Write dataset rows aligned to an averaged DeepSHAP partition."""

    from transcriptml.data.bundle import load_bundle

    indices = np.load(output_root / f"{partition}_indices.npy", allow_pickle=False)
    expected_ids = (output_root / f"{partition}_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    bundle = load_bundle(dataset, mmap_mode="r")
    if np.any(indices < 0) or np.any(indices >= bundle.X.shape[0]):
        raise ValueError("Averaged attribution indices fall outside the dataset")
    selected_ids = [str(bundle.ids[int(index)]) for index in indices]
    if selected_ids != expected_ids:
        raise ValueError("Averaged attribution IDs do not match the dataset")

    channels = int(bundle.X.shape[1]) if all_channels else 4
    shape = (len(indices), channels, int(bundle.X.shape[2]))
    sequence_path = output_root / "sequences.npy"
    temporary_sequence = output_root / ".sequences.partial.npy"
    if len(indices) == 0:
        np.save(temporary_sequence, np.empty(shape, dtype=bundle.X.dtype))
    else:
        sequence_output = np.lib.format.open_memmap(
            temporary_sequence,
            mode="w+",
            dtype=bundle.X.dtype,
            shape=shape,
        )
        try:
            for start in range(0, len(indices), int(batch_size)):
                stop = min(start + int(batch_size), len(indices))
                sequence_output[start:stop] = bundle.X[indices[start:stop], :channels]
            _close_memmap(sequence_output)
        except BaseException:
            _close_memmap(sequence_output)
            temporary_sequence.unlink(missing_ok=True)
            raise
    os.replace(temporary_sequence, sequence_path)

    index_path = output_root / "sequence_indices.npy"
    np.save(index_path, indices.astype(np.int64, copy=False))
    ids_path = output_root / "ids.txt"
    ids_path.write_text(
        "".join(f"{identifier}\n" for identifier in selected_ids),
        encoding="utf-8",
    )
    outputs = {
        "sequences": str(sequence_path),
        "sequence_indices": str(index_path),
        "ids": str(ids_path),
    }
    if bundle.metadata is not None:
        metadata_path = output_root / "metadata.json"
        metadata_path.write_text(
            json.dumps([bundle.metadata[int(index)] for index in indices], indent=2) + "\n",
            encoding="utf-8",
        )
        outputs["metadata"] = str(metadata_path)
    return outputs


def _average_deep_shap_partition(
    input_dir: str | Path,
    *,
    out_dir: str | Path | None,
    confidence: ConfidenceSubset,
    use_all_folds: bool,
    dataset: str | Path,
    alignment: AlignmentMode,
    mode_threshold: float,
    all_channels: bool,
    batch_size: int,
    dtype: np.dtype,
    progress: bool,
) -> dict[str, Any]:
    """Average actual and hypothetical DeepSHAP tensors and slice their inputs."""

    if use_all_folds and confidence != "all":
        return _average_selected_confidence_across_all_folds(
            input_dir,
            out_dir=out_dir,
            confidence=cast(Literal["high", "low"], confidence),
            dataset=dataset,
            alignment=alignment,
            mode_threshold=mode_threshold,
            all_channels=all_channels,
            batch_size=batch_size,
            dtype=dtype,
            progress=progress,
        )

    if confidence == "all":
        return _average_all_deep_shap_partitions(
            input_dir,
            out_dir=out_dir,
            dataset=dataset,
            alignment=alignment,
            mode_threshold=mode_threshold,
            all_channels=all_channels,
            batch_size=batch_size,
            dtype=dtype,
            progress=progress,
        )

    output_root = Path(input_dir) if out_dir is None else Path(out_dir)
    partition = f"{confidence}_confidence"
    summaries: dict[str, dict[str, Any]] = {}
    for attribution_kind in ("hypothetical", "actual"):
        summaries[attribution_kind] = average_attributions(
            input_dir,
            input_file=f"{partition}_{attribution_kind}_attributions.npy",
            out_dir=output_root,
            ism=False,
            method="mean",
            alignment=alignment,
            mode_threshold=mode_threshold,
            batch_size=batch_size,
            dtype=dtype,
            progress=progress,
        )

    hypothetical_rows = summaries["hypothetical"]["aligned_row_count"]
    actual_rows = summaries["actual"]["aligned_row_count"]
    if hypothetical_rows != actual_rows:
        raise RuntimeError("Actual and hypothetical averaged DeepSHAP rows do not align")
    sequence_outputs = _write_sequence_slice(
        dataset,
        output_root,
        partition=partition,
        all_channels=all_channels,
        batch_size=batch_size,
    )
    outputs = {
        "hypothetical_attributions": summaries["hypothetical"]["outputs"]["combined"],
        "actual_attributions": summaries["actual"]["outputs"]["combined"],
        "fold_support": summaries["hypothetical"]["outputs"]["fold_support"],
        "confidence_support": summaries["hypothetical"]["outputs"][
            "confidence_support"
        ],
        **sequence_outputs,
    }
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "deep_shap",
        "confidence": confidence,
        "use_all_folds": False,
        "input_dir": str(input_dir),
        "out_dir": str(output_root),
        "dataset": str(dataset),
        "fold_count": summaries["hypothetical"]["fold_count"],
        "alignment": alignment,
        "mode_threshold": mode_threshold,
        "aligned_row_count": int(hypothetical_rows),
        "fold_support_min": summaries["hypothetical"]["fold_support_min"],
        "fold_support_max": summaries["hypothetical"]["fold_support_max"],
        "all_channels": bool(all_channels),
        "sequence_shape": list(np.load(sequence_outputs["sequences"], mmap_mode="r").shape),
        "attribution_shape": summaries["hypothetical"]["shape"],
        "dtype": str(dtype),
        "batch_size": int(batch_size),
        "outputs": outputs,
    }
    (output_root / "average_attributions_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_union_average(
    unions: list[_FoldPartitionUnion],
    positions: list[np.ndarray],
    output_path: Path,
    *,
    shape_tail: tuple[int, int],
    dtype: np.dtype,
    batch_size: int,
    progress: bool,
) -> None:
    """Stream a fold mean from partition-union views to an atomic output."""

    n_rows = len(positions[0])
    shape = (n_rows, *shape_tail)
    temporary = output_path.with_name(f".{output_path.name}.partial.npy")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    reporter = ProgressReporter(
        "average-attributions",
        total=n_rows,
        unit="sequences",
        enabled=progress,
    )
    try:
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            combined = np.zeros((stop - start, *shape_tail), dtype=dtype)
            counts = np.zeros(stop - start, dtype=np.int64)
            for union, fold_positions in zip(unions, positions, strict=True):
                selected = fold_positions[start:stop]
                present = selected >= 0
                if np.any(present):
                    combined[present] += union.take(selected[present], dtype)
                    counts[present] += 1
            if np.any(counts == 0):
                raise RuntimeError("Selected DeepSHAP row has no contributing folds")
            combined /= counts[:, None, None]
            output[start:stop] = combined
            reporter.update(advance=stop - start)
        _close_memmap(output)
        os.replace(temporary, output_path)
    except BaseException:
        _close_memmap(output)
        temporary.unlink(missing_ok=True)
        raise
    reporter.close(extra=f"{len(unions)} folds; method=mean")


def _align_unions_for_confidence(
    unions: list[_FoldPartitionUnion],
    *,
    confidence: Literal["high", "low"],
    alignment: AlignmentMode,
    mode_threshold: float,
) -> tuple[np.ndarray, list[str], list[np.ndarray], np.ndarray, np.ndarray]:
    """Select by confidence membership but locate rows in either partition."""

    desired_indices = [
        union.indices[
            union.source_is_high
            if confidence == "high"
            else ~union.source_is_high
        ]
        for union in unions
    ]
    if alignment == "exact":
        selected_indices = desired_indices[0]
        for union, indices in zip(unions[1:], desired_indices[1:], strict=True):
            if not np.array_equal(indices, selected_indices):
                raise ValueError(
                    f"{confidence}-confidence DeepSHAP rows are not identically "
                    f"aligned across folds; mismatch at {union.fold_dir}"
                )
        confidence_support = np.full(
            len(selected_indices), len(unions), dtype=np.int64
        )
    else:
        selected_indices, confidence_support = _select_indices_by_fold_support(
            desired_indices,
            alignment=alignment,
            mode_threshold=mode_threshold,
        )

    positions: list[np.ndarray] = []
    ids_by_fold: list[dict[int, str]] = []
    for union in unions:
        position_by_index = {
            int(index): row for row, index in enumerate(union.indices)
        }
        positions.append(
            np.asarray(
                [position_by_index.get(int(index), -1) for index in selected_indices],
                dtype=np.int64,
            )
        )
        ids_by_fold.append(
            {int(index): union.ids[row] for row, index in enumerate(union.indices)}
        )

    selected_ids: list[str] = []
    for index in selected_indices:
        observed = [
            values[int(index)] for values in ids_by_fold if int(index) in values
        ]
        if any(identifier != observed[0] for identifier in observed[1:]):
            raise ValueError(f"Sequence ID mismatch for global index {int(index)}")
        selected_ids.append(observed[0])
    contribution_support = np.sum(
        np.stack([fold_positions >= 0 for fold_positions in positions], axis=0),
        axis=0,
        dtype=np.int64,
    )
    return (
        selected_indices,
        selected_ids,
        positions,
        confidence_support,
        contribution_support,
    )


def _average_selected_confidence_across_all_folds(
    input_dir: str | Path,
    *,
    out_dir: str | Path | None,
    confidence: Literal["high", "low"],
    dataset: str | Path,
    alignment: AlignmentMode,
    mode_threshold: float,
    all_channels: bool,
    batch_size: int,
    dtype: np.dtype,
    progress: bool,
) -> dict[str, Any]:
    """Select by confidence support, then average every eligible fold."""

    output_root = Path(input_dir) if out_dir is None else Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    kind_data: dict[str, tuple[Any, ...]] = {}
    for attribution_kind in ("hypothetical", "actual"):
        kind = cast(Literal["actual", "hypothetical"], attribution_kind)
        unions, paths, shape_tail = _load_deep_shap_unions(input_dir, kind)
        alignment_result = _align_unions_for_confidence(
            unions,
            confidence=confidence,
            alignment=alignment,
            mode_threshold=mode_threshold,
        )
        kind_data[attribution_kind] = (
            unions,
            paths,
            shape_tail,
            *alignment_result,
        )

    hypothetical = kind_data["hypothetical"]
    actual = kind_data["actual"]
    if (
        len(hypothetical[0]) != len(actual[0])
        or hypothetical[2] != actual[2]
        or not np.array_equal(hypothetical[3], actual[3])
        or hypothetical[4] != actual[4]
        or not np.array_equal(hypothetical[6], actual[6])
        or not np.array_equal(hypothetical[7], actual[7])
    ):
        raise ValueError(
            "Actual and hypothetical DeepSHAP rows do not align for all-fold averaging"
        )

    partition = f"{confidence}_confidence"
    attribution_outputs: dict[str, str] = {}
    for attribution_kind in ("hypothetical", "actual"):
        unions, _, shape_tail, _, _, positions, _, _ = kind_data[attribution_kind]
        output_path = (
            output_root
            / f"average_{partition}_{attribution_kind}_attributions.npy"
        )
        _write_union_average(
            unions,
            positions,
            output_path,
            shape_tail=shape_tail,
            dtype=dtype,
            batch_size=batch_size,
            progress=progress,
        )
        attribution_outputs[f"{attribution_kind}_attributions"] = str(output_path)

    indices = hypothetical[3]
    ids = hypothetical[4]
    confidence_support = hypothetical[6]
    fold_support = hypothetical[7]
    indices_path = output_root / f"{partition}_indices.npy"
    ids_path = output_root / f"{partition}_ids.txt"
    confidence_support_path = output_root / f"{partition}_confidence_support.npy"
    fold_support_path = output_root / f"{partition}_fold_support.npy"
    np.save(indices_path, indices)
    np.save(confidence_support_path, confidence_support)
    np.save(fold_support_path, fold_support)
    ids_path.write_text(
        "".join(f"{identifier}\n" for identifier in ids),
        encoding="utf-8",
    )
    sequence_outputs = _write_sequence_slice(
        dataset,
        output_root,
        partition=partition,
        all_channels=all_channels,
        batch_size=batch_size,
    )
    outputs = {
        **attribution_outputs,
        "indices": str(indices_path),
        "partition_ids": str(ids_path),
        "confidence_support": str(confidence_support_path),
        "fold_support": str(fold_support_path),
        **sequence_outputs,
    }
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "deep_shap",
        "confidence": confidence,
        "use_all_folds": True,
        "input_dir": str(input_dir),
        "out_dir": str(output_root),
        "dataset": str(dataset),
        "fold_count": len(hypothetical[0]),
        "fold_files": {
            kind: [str(path) for path in data[1]] for kind, data in kind_data.items()
        },
        "alignment": alignment,
        "mode_threshold": mode_threshold,
        "aligned_row_count": int(len(indices)),
        "confidence_support_min": (
            int(np.min(confidence_support)) if len(confidence_support) else None
        ),
        "confidence_support_max": (
            int(np.max(confidence_support)) if len(confidence_support) else None
        ),
        "fold_support_min": int(np.min(fold_support)) if len(fold_support) else None,
        "fold_support_max": int(np.max(fold_support)) if len(fold_support) else None,
        "all_channels": bool(all_channels),
        "sequence_shape": list(
            np.load(sequence_outputs["sequences"], mmap_mode="r").shape
        ),
        "attribution_shape": [len(indices), *hypothetical[2]],
        "dtype": str(dtype),
        "batch_size": int(batch_size),
        "outputs": outputs,
    }
    (output_root / "average_attributions_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _average_all_deep_shap_partitions(
    input_dir: str | Path,
    *,
    out_dir: str | Path | None,
    dataset: str | Path,
    alignment: AlignmentMode,
    mode_threshold: float,
    all_channels: bool,
    batch_size: int,
    dtype: np.dtype,
    progress: bool,
) -> dict[str, Any]:
    """Average the union of eligible high- and low-confidence rows per fold."""

    output_root = Path(input_dir) if out_dir is None else Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    kind_data: dict[
        str,
        tuple[
            list[_FoldPartitionUnion],
            list[Path],
            tuple[int, int],
            np.ndarray,
            list[str],
            list[np.ndarray],
            np.ndarray,
        ],
    ] = {}
    for attribution_kind in ("hypothetical", "actual"):
        kind = cast(Literal["actual", "hypothetical"], attribution_kind)
        unions, paths, shape_tail = _load_deep_shap_unions(input_dir, kind)
        indices, ids, positions, support = _align_partition_unions(
            unions,
            alignment,
            mode_threshold,
        )
        kind_data[attribution_kind] = (
            unions,
            paths,
            shape_tail,
            indices,
            ids,
            positions,
            support,
        )

    hypothetical = kind_data["hypothetical"]
    actual = kind_data["actual"]
    if (
        len(hypothetical[0]) != len(actual[0])
        or hypothetical[2] != actual[2]
        or not np.array_equal(hypothetical[3], actual[3])
        or hypothetical[4] != actual[4]
        or not np.array_equal(hypothetical[6], actual[6])
    ):
        raise ValueError(
            "Actual and hypothetical DeepSHAP rows do not align after combining "
            "high- and low-confidence partitions"
        )

    attribution_outputs: dict[str, str] = {}
    for attribution_kind in ("hypothetical", "actual"):
        unions, _, shape_tail, _, _, positions, _ = kind_data[attribution_kind]
        output_path = output_root / f"average_all_{attribution_kind}_attributions.npy"
        _write_union_average(
            unions,
            positions,
            output_path,
            shape_tail=shape_tail,
            dtype=dtype,
            batch_size=batch_size,
            progress=progress,
        )
        attribution_outputs[f"{attribution_kind}_attributions"] = str(output_path)

    indices = hypothetical[3]
    ids = hypothetical[4]
    indices_path = output_root / "all_indices.npy"
    ids_path = output_root / "all_ids.txt"
    support_path = output_root / "all_fold_support.npy"
    np.save(indices_path, indices)
    np.save(support_path, hypothetical[6])
    ids_path.write_text(
        "".join(f"{identifier}\n" for identifier in ids),
        encoding="utf-8",
    )
    sequence_outputs = _write_sequence_slice(
        dataset,
        output_root,
        partition="all",
        all_channels=all_channels,
        batch_size=batch_size,
    )
    outputs = {
        **attribution_outputs,
        "indices": str(indices_path),
        "partition_ids": str(ids_path),
        "fold_support": str(support_path),
        **sequence_outputs,
    }
    fold_row_counts = [len(union.indices) for union in hypothetical[0]]
    attribution_shape = [len(indices), *hypothetical[2]]
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "deep_shap",
        "confidence": "all",
        "input_dir": str(input_dir),
        "out_dir": str(output_root),
        "dataset": str(dataset),
        "fold_count": len(hypothetical[0]),
        "fold_files": {
            kind: [str(path) for path in data[1]] for kind, data in kind_data.items()
        },
        "fold_row_counts": fold_row_counts,
        "alignment": alignment,
        "mode_threshold": mode_threshold,
        "aligned_row_count": int(len(indices)),
        "fold_support_min": (
            int(np.min(hypothetical[6])) if len(hypothetical[6]) else None
        ),
        "fold_support_max": (
            int(np.max(hypothetical[6])) if len(hypothetical[6]) else None
        ),
        "all_channels": bool(all_channels),
        "sequence_shape": list(np.load(sequence_outputs["sequences"], mmap_mode="r").shape),
        "attribution_shape": attribution_shape,
        "dtype": str(dtype),
        "batch_size": int(batch_size),
        "outputs": outputs,
    }
    (output_root / "average_attributions_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_average_attributions_from_args(args: Any) -> dict[str, Any]:
    """Run attribution averaging from parsed CLI arguments."""

    return average_attributions(
        args.input_dir,
        input_file=args.input_file,
        out_dir=args.out_dir,
        ism=args.ism,
        method=args.method,
        alignment=args.alignment,
        mode_threshold=args.mode_threshold,
        confidence=args.confidence,
        use_all_folds=args.use_all_folds,
        dataset=args.dataset,
        all_channels=args.all_channels,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
