from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _fold_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"fold(\d+)", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def find_delta_paths(input_dir: str | Path) -> list[Path]:
    """Return fold-level ``deltas.npy`` paths in natural fold order."""

    root = Path(input_dir)
    paths = [
        fold_dir / "deltas.npy"
        for fold_dir in root.iterdir()
        if fold_dir.is_dir() and fold_dir.name.startswith("fold") and (fold_dir / "deltas.npy").exists()
    ]
    return sorted(paths, key=lambda path: _fold_sort_key(path.parent))


def _validate_arrays(delta_paths: list[Path]) -> tuple[int, int, int]:
    if not delta_paths:
        raise FileNotFoundError("No fold*/deltas.npy files found")

    ref = np.load(delta_paths[0], mmap_mode="r", allow_pickle=False)
    if ref.ndim != 3:
        raise ValueError(f"Expected {delta_paths[0]} to have shape (N, 4, L); got {ref.shape}")
    n_examples, n_channels, length = (int(x) for x in ref.shape)
    if n_channels != 4:
        raise ValueError(f"Expected ISM channel dimension to be 4; got shape {ref.shape}")

    expected = (n_examples, n_channels, length)
    for path in delta_paths[1:]:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != expected:
            raise ValueError(f"Shape mismatch for {path}: {arr.shape} != {expected}")
    return expected


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def summarize_ism_folds(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 256,
    dtype: str | np.dtype = "float32",
    dataset: str | Path | None = None,
    write_fold_std: bool = False,
    write_projected: bool = False,
) -> dict[str, Any]:
    """Average fold-level single-nucleotide ISM arrays and write summary arrays.

    Args:
        input_dir: Directory containing ``fold*/deltas.npy`` outputs.
        out_dir: Directory where summary arrays and ``summary.json`` are written.
        batch_size: Number of sequences processed per chunk.
        dtype: Floating-point dtype used for accumulation and output arrays.
        dataset: Optional dataset bundle used to validate sequence order, copy
            IDs, and provide one-hot reference bases for projected scores.
        write_fold_std: Whether to write fold-wise standard deviation of raw
            deltas.
        write_projected: Whether to write mean-centered scores multiplied by
            reference one-hot bases. Requires ``dataset``.
    """

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype must be a floating-point dtype, got {dtype}")

    delta_paths = find_delta_paths(input_dir)
    if not delta_paths:
        raise FileNotFoundError(f"No fold*/deltas.npy files found in {input_dir}")
    n_examples, n_channels, length = _validate_arrays(delta_paths)
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in delta_paths]
    n_folds = len(arrays)

    bundle = None
    if dataset is not None:
        from transcriptml.data.bundle import load_bundle

        bundle = load_bundle(dataset, mmap_mode="r")
        if int(bundle.X.shape[0]) != n_examples or int(bundle.X.shape[2]) != length:
            raise ValueError(
                "Dataset X must match ISM arrays in N and L. "
                f"Got X={bundle.X.shape}, ISM={(n_examples, n_channels, length)}"
            )
        if int(bundle.X.shape[1]) < 4:
            raise ValueError(f"Dataset X must have at least four base channels; got {bundle.X.shape}")
        (out / "ids.txt").write_text("\n".join(str(x) for x in bundle.ids) + "\n", encoding="utf-8")
    elif write_projected:
        raise ValueError("write_projected requires --dataset so reference bases can be loaded")

    avg_path = out / "average_deltas.npy"
    centered_path = out / "average_mean_centered_deltas.npy"
    avg_out = np.lib.format.open_memmap(avg_path, mode="w+", dtype=dtype, shape=(n_examples, n_channels, length))
    centered_out = np.lib.format.open_memmap(
        centered_path,
        mode="w+",
        dtype=dtype,
        shape=(n_examples, n_channels, length),
    )
    std_out = None
    std_path = out / "fold_std_deltas.npy"
    if write_fold_std:
        std_out = np.lib.format.open_memmap(std_path, mode="w+", dtype=dtype, shape=(n_examples, n_channels, length))
    projected_out = None
    projected_path = out / "average_projected_mean_centered_deltas.npy"
    if write_projected:
        projected_out = np.lib.format.open_memmap(
            projected_path,
            mode="w+",
            dtype=dtype,
            shape=(n_examples, n_channels, length),
        )

    for start in range(0, n_examples, int(batch_size)):
        end = min(start + int(batch_size), n_examples)
        chunk = np.zeros((end - start, n_channels, length), dtype=dtype)
        sumsq = np.zeros_like(chunk) if std_out is not None else None

        for arr in arrays:
            values = arr[start:end].astype(dtype, copy=False)
            chunk += values
            if sumsq is not None:
                sumsq += values * values

        chunk /= n_folds
        avg_out[start:end] = chunk

        centered = chunk - chunk.mean(axis=1, keepdims=True)
        centered_out[start:end] = centered

        if std_out is not None and sumsq is not None:
            variance = (sumsq / n_folds) - (chunk * chunk)
            np.maximum(variance, 0, out=variance)
            np.sqrt(variance, out=variance)
            std_out[start:end] = variance

        if projected_out is not None:
            assert bundle is not None
            reference = np.asarray(bundle.X[start:end, :4, :], dtype=dtype)
            projected_out[start:end] = centered * reference

        print(f"summarize-ism: processed {end:,} / {n_examples:,}")

    for arr in (avg_out, centered_out, std_out, projected_out):
        if arr is not None:
            arr.flush()

    outputs = {
        "average_deltas": str(avg_path),
        "average_mean_centered_deltas": str(centered_path),
    }
    if std_out is not None:
        outputs["fold_std_deltas"] = str(std_path)
    if projected_out is not None:
        outputs["average_projected_mean_centered_deltas"] = str(projected_path)

    summary: dict[str, Any] = {
        "analysis": "single_nucleotide_ism_summary",
        "input_dir": str(input_dir),
        "out_dir": str(out),
        "delta_definition": "mutant_prediction - reference_prediction",
        "fold_count": n_folds,
        "fold_delta_files": [str(path) for path in delta_paths],
        "shape": [n_examples, n_channels, length],
        "dtype": str(dtype),
        "batch_size": int(batch_size),
        "mean_centered_axis": "channels",
        "same_sequence_order_required": True,
        "dataset": str(dataset) if dataset is not None else None,
        "ids_written": dataset is not None,
        "outputs": outputs,
    }
    _write_json(out / "summary.json", summary)
    print(f"summarize-ism: wrote {out / 'summary.json'}")
    return summary


def run_ism_summary_from_args(args: Any) -> dict[str, Any]:
    """Run ISM summarization from parsed CLI args."""

    return summarize_ism_folds(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        dtype=args.dtype,
        dataset=args.dataset,
        write_fold_std=args.write_fold_std,
        write_projected=args.write_projected,
    )
