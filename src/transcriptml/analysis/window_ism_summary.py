from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from transcriptml.interpret.window_ism import sequence_ids_sha256


_ARRAY_NAMES = (
    "window_starts",
    "window_mask",
    "mean_deltas",
    "mean_abs_deltas",
    "std_deltas",
    "reference_predictions",
    "valid_lengths",
)
_MATCHED_SETTINGS = ("input_shape", "window_size", "stride", "n_ablations", "seed", "mutation_policy")


def _fold_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"fold(\d+)", path.name)
    if match:
        return 0, int(match.group(1))
    return 1, path.name


def find_window_ism_fold_dirs(input_dir: str | Path) -> list[Path]:
    """Find complete ``fold*`` window-ISM result directories."""

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Window-ISM input directory does not exist: {root}")
    fold_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("fold")
        and (path / "summary.json").exists()
        and all((path / f"{name}.npy").exists() for name in _ARRAY_NAMES)
    ]
    return sorted(fold_dirs, key=_fold_sort_key)


def _load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("analysis") != "window_ism":
        raise ValueError(f"Expected a window_ism summary at {path}")
    return value


def _validate_fold_inputs(fold_dirs: list[Path]) -> tuple[dict[str, Any], tuple[int, int]]:
    if not fold_dirs:
        raise FileNotFoundError("No complete fold*/ window-ISM outputs found")

    reference_summary = _load_summary(fold_dirs[0] / "summary.json")
    reference_arrays = {
        name: np.load(fold_dirs[0] / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in _ARRAY_NAMES
    }
    effect_shape = tuple(int(value) for value in reference_arrays["mean_deltas"].shape)
    if len(effect_shape) != 2:
        raise ValueError(f"Expected window effects with shape (N, Wmax), got {effect_shape}")
    for name in ("window_starts", "window_mask", "mean_abs_deltas", "std_deltas"):
        if reference_arrays[name].shape != effect_shape:
            raise ValueError(f"{fold_dirs[0] / f'{name}.npy'} shape does not match mean_deltas.npy")
    if reference_arrays["valid_lengths"].shape != (effect_shape[0],):
        raise ValueError("valid_lengths.npy must have shape (N,)")
    if reference_arrays["reference_predictions"].shape != (effect_shape[0],):
        raise ValueError("reference_predictions.npy must have shape (N,)")

    for fold_dir in fold_dirs[1:]:
        summary = _load_summary(fold_dir / "summary.json")
        for setting in _MATCHED_SETTINGS:
            if summary.get(setting) != reference_summary.get(setting):
                raise ValueError(
                    f"Window-ISM setting mismatch for {setting}: "
                    f"{fold_dir} has {summary.get(setting)!r}, expected {reference_summary.get(setting)!r}"
                )
        ref_ids = reference_summary.get("sequence_ids_sha256")
        fold_ids = summary.get("sequence_ids_sha256")
        if fold_ids != ref_ids:
            raise ValueError(f"Sequence ID ordering mismatch for {fold_dir}")

        arrays = {
            name: np.load(fold_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in _ARRAY_NAMES
        }
        for name in ("mean_deltas", "mean_abs_deltas", "std_deltas", "reference_predictions"):
            if arrays[name].shape != reference_arrays[name].shape:
                raise ValueError(
                    f"Shape mismatch for {fold_dir / f'{name}.npy'}: "
                    f"{arrays[name].shape} != {reference_arrays[name].shape}"
                )
        for name in ("window_starts", "window_mask", "valid_lengths"):
            if not np.array_equal(arrays[name], reference_arrays[name]):
                raise ValueError(f"Coordinate or mask mismatch for {fold_dir / f'{name}.npy'}")
    return reference_summary, effect_shape


def summarize_window_ism_folds(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    dataset: str | Path | None = None,
    batch_size: int = 256,
    dtype: str | np.dtype = "float32",
) -> dict[str, Any]:
    """Aggregate matching fold-level window-ISM tracks.

    Signed means and mean absolute effects are averaged across models. Random-
    mutation variability is summarized as the root mean within-model variance,
    while model disagreement is the population standard deviation of fold-level
    signed means.
    """

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError(f"dtype must be floating-point, got {output_dtype}")

    fold_dirs = find_window_ism_fold_dirs(input_dir)
    reference_summary, effect_shape = _validate_fold_inputs(fold_dirs)
    n_sequences, max_windows = effect_shape
    n_folds = len(fold_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    starts = np.load(fold_dirs[0] / "window_starts.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(fold_dirs[0] / "window_mask.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(fold_dirs[0] / "valid_lengths.npy", mmap_mode="r", allow_pickle=False)
    np.save(out / "window_starts.npy", starts)
    np.save(out / "window_mask.npy", mask)
    np.save(out / "valid_lengths.npy", lengths)

    if dataset is not None:
        from transcriptml.data.bundle import load_bundle

        bundle = load_bundle(dataset, mmap_mode="r")
        if int(bundle.X.shape[0]) != n_sequences:
            raise ValueError(f"Dataset has N={bundle.X.shape[0]}, expected N={n_sequences}")
        expected_input_shape = tuple(int(value) for value in reference_summary["input_shape"])
        if tuple(int(value) for value in bundle.X.shape) != expected_input_shape:
            raise ValueError(
                f"Dataset X shape {bundle.X.shape} does not match saved input shape {expected_input_shape}"
            )
        ids_digest = sequence_ids_sha256(bundle.ids)
        saved_digest = reference_summary.get("sequence_ids_sha256")
        if saved_digest is not None and ids_digest != saved_digest:
            raise ValueError("Dataset sequence ID ordering does not match the window-ISM fold outputs")
        (out / "ids.txt").write_text("\n".join(str(value) for value in bundle.ids) + "\n", encoding="utf-8")

    output_specs = {
        "average_mean_deltas": effect_shape,
        "average_mean_abs_deltas": effect_shape,
        "within_model_std_deltas": effect_shape,
        "fold_std_mean_deltas": effect_shape,
        "average_reference_predictions": (n_sequences,),
    }
    outputs = {
        name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=output_dtype, shape=shape)
        for name, shape in output_specs.items()
    }
    fold_means = [np.load(path / "mean_deltas.npy", mmap_mode="r", allow_pickle=False) for path in fold_dirs]
    fold_abs = [np.load(path / "mean_abs_deltas.npy", mmap_mode="r", allow_pickle=False) for path in fold_dirs]
    fold_std = [np.load(path / "std_deltas.npy", mmap_mode="r", allow_pickle=False) for path in fold_dirs]

    for start in range(0, n_sequences, int(batch_size)):
        end = min(start + int(batch_size), n_sequences)
        chunk_shape = (end - start, max_windows)
        mean_sum = np.zeros(chunk_shape, dtype=np.float64)
        mean_sumsq = np.zeros(chunk_shape, dtype=np.float64)
        abs_sum = np.zeros(chunk_shape, dtype=np.float64)
        within_variance_sum = np.zeros(chunk_shape, dtype=np.float64)
        for means, magnitudes, deviations in zip(fold_means, fold_abs, fold_std):
            mean_values = np.asarray(means[start:end], dtype=np.float64)
            std_values = np.asarray(deviations[start:end], dtype=np.float64)
            mean_sum += mean_values
            mean_sumsq += mean_values * mean_values
            abs_sum += np.asarray(magnitudes[start:end], dtype=np.float64)
            within_variance_sum += std_values * std_values

        average_mean = mean_sum / n_folds
        between_variance = (mean_sumsq / n_folds) - (average_mean * average_mean)
        np.maximum(between_variance, 0.0, out=between_variance)
        outputs["average_mean_deltas"][start:end] = average_mean
        outputs["average_mean_abs_deltas"][start:end] = abs_sum / n_folds
        outputs["within_model_std_deltas"][start:end] = np.sqrt(within_variance_sum / n_folds)
        outputs["fold_std_mean_deltas"][start:end] = np.sqrt(between_variance)

    ref_sum = np.zeros(n_sequences, dtype=np.float64)
    for fold_dir in fold_dirs:
        ref_sum += np.load(fold_dir / "reference_predictions.npy", mmap_mode="r", allow_pickle=False)
    outputs["average_reference_predictions"][:] = ref_sum / n_folds
    for values in outputs.values():
        values.flush()

    summary = {
        "analysis": "window_ism_summary",
        "input_dir": str(input_dir),
        "out_dir": str(out),
        "fold_count": n_folds,
        "fold_dirs": [str(path) for path in fold_dirs],
        "effect_definition": "mutant_prediction - reference_prediction",
        "window_size": reference_summary["window_size"],
        "stride": reference_summary["stride"],
        "n_ablations": reference_summary["n_ablations"],
        "seed": reference_summary["seed"],
        "mutation_policy": reference_summary["mutation_policy"],
        "shape": [n_sequences, max_windows],
        "dtype": str(output_dtype),
        "batch_size": int(batch_size),
        "same_sequence_order_required": True,
        "dataset": str(dataset) if dataset is not None else None,
        "ids_written": dataset is not None,
        "within_model_std_definition": "sqrt(mean_folds(std_deltas ** 2))",
        "fold_std_definition": "population std across fold mean_deltas",
        "outputs": {
            **{name: str(out / f"{name}.npy") for name in output_specs},
            "window_starts": str(out / "window_starts.npy"),
            "window_mask": str(out / "window_mask.npy"),
            "valid_lengths": str(out / "valid_lengths.npy"),
            "ids": str(out / "ids.txt") if dataset is not None else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_window_ism_summary_from_args(args: Any) -> dict[str, Any]:
    """Run fold-level window-ISM aggregation from parsed CLI arguments."""

    return summarize_window_ism_folds(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
