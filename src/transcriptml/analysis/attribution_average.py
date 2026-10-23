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
AlignmentMode = Literal["intersection", "exact"]
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
) -> tuple[str, np.ndarray, list[str], list[np.ndarray]] | None:
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
        return partition, reference_indices, reference_ids, positions

    common = set(int(index) for index in fold_indices[0])
    for indices in fold_indices[1:]:
        common.intersection_update(int(index) for index in indices)
    common_indices = np.asarray(sorted(common), dtype=np.int64)
    positions: list[np.ndarray] = []
    common_ids: list[str] = []
    for fold_number, (indices, ids) in enumerate(zip(fold_indices, fold_ids, strict=True)):
        position_by_index = {int(index): row for row, index in enumerate(indices)}
        positions.append(
            np.asarray([position_by_index[int(index)] for index in common_indices], dtype=np.int64)
        )
        current_ids = {int(index): ids[row] for row, index in enumerate(indices)}
        if fold_number == 0:
            common_ids = [current_ids[int(index)] for index in common_indices]
        else:
            for index, identifier in zip(common_indices, common_ids, strict=True):
                if current_ids[int(index)] != identifier:
                    raise ValueError(
                        f"Sequence ID mismatch for global index {int(index)} in {paths[fold_number].parent}"
                    )

    return partition, common_indices, common_ids, positions


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


def average_attributions(
    input_dir: str | Path,
    *,
    input_file: str | Path | None = None,
    out_dir: str | Path | None = None,
    ism: bool = False,
    method: CombinationMethod = "mean",
    alignment: AlignmentMode = "intersection",
    confidence: Literal["high", "low"] | None = None,
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
    alignment_value = str(alignment).strip().lower()
    if alignment_value not in {"intersection", "exact"}:
        raise ValueError("alignment must be either 'intersection' or 'exact'")
    selected_alignment = cast(AlignmentMode, alignment_value)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError(f"dtype must be floating-point, got {output_dtype}")

    if not ism and input_file is None:
        if confidence not in {"high", "low"}:
            raise ValueError("DeepSHAP averaging requires confidence='high' or 'low'")
        if dataset is None:
            raise ValueError("DeepSHAP averaging requires a dataset for aligned sequences")
        return _average_deep_shap_partition(
            input_dir,
            out_dir=out_dir,
            confidence=confidence,
            dataset=dataset,
            alignment=selected_alignment,
            all_channels=all_channels,
            batch_size=int(batch_size),
            dtype=output_dtype,
            progress=progress,
        )

    if input_file is None and not ism:
        raise ValueError("input_file is required for DeepSHAP attribution averaging")
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
        partition, indices, ids, _ = alignment_result
        indices_output = output_root / f"{partition}_indices.npy"
        ids_output = output_root / f"{partition}_ids.txt"
        np.save(indices_output, indices)
        ids_output.write_text(
            "".join(f"{identifier}\n" for identifier in ids),
            encoding="utf-8",
        )
        outputs.update(indices=str(indices_output), ids=str(ids_output))
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "ism" if ism else "deep_shap",
        "input_dir": str(input_dir),
        "input_file": str(relative),
        "fold_count": len(paths),
        "fold_files": [str(path) for path in paths],
        "combination_method": selected_method,
        "alignment": selected_alignment if partitioned else "row_order",
        "fold_row_counts": [int(array.shape[0]) for array in arrays],
        "aligned_row_count": int(shape[0]),
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
    confidence: Literal["high", "low"],
    dataset: str | Path,
    alignment: AlignmentMode,
    all_channels: bool,
    batch_size: int,
    dtype: np.dtype,
    progress: bool,
) -> dict[str, Any]:
    """Average actual and hypothetical DeepSHAP tensors and slice their inputs."""

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
        **sequence_outputs,
    }
    summary: dict[str, Any] = {
        "analysis": "average_attributions",
        "attribution_type": "deep_shap",
        "confidence": confidence,
        "input_dir": str(input_dir),
        "out_dir": str(output_root),
        "dataset": str(dataset),
        "fold_count": summaries["hypothetical"]["fold_count"],
        "alignment": alignment,
        "aligned_row_count": int(hypothetical_rows),
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


def run_average_attributions_from_args(args: Any) -> dict[str, Any]:
    """Run attribution averaging from parsed CLI arguments."""

    return average_attributions(
        args.input_dir,
        input_file=args.input_file,
        out_dir=args.out_dir,
        ism=args.ism,
        method=args.method,
        alignment=args.alignment,
        confidence=args.confidence,
        dataset=args.dataset,
        all_channels=args.all_channels,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
