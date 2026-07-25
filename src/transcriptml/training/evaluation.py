from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from transcriptml.data.bundle import DatasetBundle, load_bundle
from transcriptml.devices import resolve_device
from transcriptml.models.common import squeeze_prediction
from transcriptml.models.registry import load_checkpoint
from transcriptml.training.metrics import mse, pearson_corr
from transcriptml.progress import ProgressReporter, log_progress


@torch.no_grad()
def predict_array(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> np.ndarray:
    """Predict scalar outputs for every example in an array.

    Args:
        model: PyTorch model that returns one scalar prediction per example.
        X: Encoded ``(N, C, L)`` input array.
        batch_size: Number of examples to score per prediction batch.
        device: Torch device used for model execution.
        progress: Whether to emit progress messages while predicting.
    """

    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    preds: list[np.ndarray] = []
    arr = X if isinstance(X, np.ndarray) else np.asarray(X)
    reporter = ProgressReporter(
        "predict array",
        total=int(arr.shape[0]),
        unit="examples",
        enabled=progress,
    )
    for start in range(0, int(arr.shape[0]), int(batch_size)):
        xb = torch.as_tensor(np.asarray(arr[start : start + int(batch_size)]), dtype=torch.float32).to(device)
        y = squeeze_prediction(model(xb))
        preds.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
        reporter.update(advance=int(xb.shape[0]))
    reporter.close()
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)


@torch.no_grad()
def _predict_indexed_array(
    model: torch.nn.Module,
    X: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress: bool = True,
    progress_label: str = "predict indexed array",
) -> np.ndarray:
    """Predict scalar outputs for selected array indices.

    Args:
        model: PyTorch model that returns one scalar prediction per example.
        X: Encoded ``(N, C, L)`` input array.
        indices: Integer indices selecting examples from ``X``.
        batch_size: Number of examples to score per prediction batch.
        device: Torch device used for model execution.
        progress: Whether to emit progress messages while predicting.
        progress_label: Label shown in progress messages.
    """

    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    preds: list[np.ndarray] = []
    reporter = ProgressReporter(
        progress_label,
        total=int(indices.shape[0]),
        unit="examples",
        enabled=progress,
    )
    for start in range(0, int(indices.shape[0]), int(batch_size)):
        batch_idx = indices[start : start + int(batch_size)]
        xb = torch.as_tensor(np.asarray(X[batch_idx]), dtype=torch.float32).to(device)
        y = squeeze_prediction(model(xb))
        preds.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
        reporter.update(advance=int(batch_idx.shape[0]))
    reporter.close()
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)


def evaluate_model(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    indices: Sequence[int] | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> dict[str, object]:
    """Evaluate a model on a dataset bundle and optional subset indices.

    Args:
        model: PyTorch model that returns one scalar prediction per example.
        bundle: Dataset bundle containing encoded inputs and optional targets.
        indices: Optional example indices to evaluate. When omitted, all
            examples are evaluated.
        batch_size: Number of examples to score per prediction batch.
        device: Torch device used for model execution.
        progress: Whether to emit progress messages while evaluating.
    """

    device = resolve_device(device)
    idx = np.arange(bundle.X.shape[0]) if indices is None else np.asarray(indices, dtype=int)
    preds = _predict_indexed_array(
        model,
        bundle.X,
        idx,
        batch_size=batch_size,
        device=device,
        progress=progress,
        progress_label="evaluate: predict",
    )
    result: dict[str, object] = {"predictions": preds, "indices": idx.tolist()}
    if bundle.y is not None:
        targets = np.asarray(bundle.y[idx], dtype=np.float32)
        result.update(
            {
                "targets": targets,
                "loss": mse(targets, preds),
                "pearson": pearson_corr(targets, preds),
            }
        )
    return result


def predict_to_csv(
    path: str | Path,
    *,
    ids: Sequence[str],
    predictions: Sequence[float],
    targets: Sequence[float] | None = None,
    indices: Sequence[int] | None = None,
) -> None:
    """Write prediction rows, and optional targets, to a CSV file.

    Args:
        path: Destination CSV path.
        ids: Example identifiers aligned to ``predictions``.
        predictions: Scalar model predictions.
        targets: Optional scalar targets aligned to ``predictions``.
        indices: Optional original dataset indices aligned to ``predictions``.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["index", "id", "prediction"]
        if targets is not None:
            fieldnames.append("target")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        idx = list(range(len(predictions))) if indices is None else list(indices)
        for j, pred in enumerate(predictions):
            row = {"index": int(idx[j]), "id": str(ids[j]), "prediction": float(pred)}
            if targets is not None:
                row["target"] = float(targets[j])
            writer.writerow(row)


def _fold_ensemble_to_csv(
    path: str | Path,
    *,
    ids: Sequence[str],
    average_predictions: Sequence[float],
    targets: Sequence[float] | None = None,
    average_residuals: Sequence[float] | None = None,
    indices: Sequence[int] | None = None,
) -> None:
    """Write per-example mean predictions and optional mean residuals."""

    if (targets is None) != (average_residuals is None):
        raise ValueError("targets and average_residuals must either both be provided or both be omitted")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["index", "id"]
        if targets is not None:
            fieldnames.append("target")
        fieldnames.append("average_prediction")
        if average_residuals is not None:
            fieldnames.append("average_residual")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        idx = list(range(len(average_predictions))) if indices is None else list(indices)
        for j, prediction in enumerate(average_predictions):
            row: dict[str, object] = {
                "index": int(idx[j]),
                "id": str(ids[j]),
                "average_prediction": float(prediction),
            }
            if targets is not None and average_residuals is not None:
                row["target"] = float(targets[j])
                row["average_residual"] = float(average_residuals[j])
            writer.writerow(row)


def evaluate_fold_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    dataset_path: str | Path,
    out_csv: str | Path | None = None,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> dict[str, object]:
    """Average predictions from fold checkpoints evaluated on one shared dataset.

    Every checkpoint scores every example in the dataset. This produces an
    ensemble prediction rather than an out-of-fold CV prediction; disjoint
    fold-level test prediction tables should be concatenated instead.

    Args:
        checkpoint_paths: Non-empty sequence of TranscriptML checkpoints.
        dataset_path: Shared dataset bundle scored by every checkpoint.
        out_csv: Optional destination for per-example ensemble predictions and
            residuals. A sibling ``.summary.json`` file is also written.
        batch_size: Number of examples to score per prediction batch.
        device: Torch device used to load and run each model.
        progress: Whether to emit progress messages while evaluating.

    Returns:
        A dictionary containing ``average_predictions``, example identifiers
        and indices, fold provenance, and, when targets are available,
        ``targets``, ``average_residuals`` (truth minus prediction), MSE,
        Pearson correlation, and mean residual.
    """

    paths = [Path(path) for path in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint does not exist: {missing[0]}")

    resolved_device = resolve_device(device)
    log_progress(f"ensemble: loading dataset {dataset_path}", enabled=progress)
    bundle = load_bundle(dataset_path, mmap_mode="r")
    indices = np.arange(int(bundle.X.shape[0]), dtype=int)
    prediction_sum = np.zeros(indices.shape[0], dtype=np.float64)

    for fold_number, checkpoint_path in enumerate(paths, start=1):
        log_progress(
            f"ensemble: loading checkpoint {fold_number}/{len(paths)}: {checkpoint_path}",
            enabled=progress,
        )
        model, _ = load_checkpoint(checkpoint_path, map_location=resolved_device)
        predictions = _predict_indexed_array(
            model,
            bundle.X,
            indices,
            batch_size=int(batch_size),
            device=resolved_device,
            progress=progress,
            progress_label=f"ensemble: checkpoint {fold_number}/{len(paths)}",
        )
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if predictions.shape != prediction_sum.shape:
            raise ValueError(
                f"Checkpoint {checkpoint_path} returned {predictions.shape[0]} predictions; "
                f"expected {prediction_sum.shape[0]}"
            )
        prediction_sum += predictions
        del model

    average_predictions64 = prediction_sum / len(paths)
    average_predictions = average_predictions64.astype(np.float32)
    result: dict[str, object] = {
        "average_predictions": average_predictions,
        "indices": indices.tolist(),
        "ids": [str(identifier) for identifier in bundle.ids],
        "fold_count": len(paths),
        "checkpoint_paths": [str(path) for path in paths],
    }

    targets = None
    average_residuals = None
    if bundle.y is not None:
        targets = np.asarray(bundle.y, dtype=np.float32).reshape(-1)
        if targets.shape != average_predictions.shape:
            raise ValueError(
                f"Dataset targets have shape {targets.shape}; expected {average_predictions.shape}"
            )
        average_residuals = (targets.astype(np.float64) - average_predictions64).astype(np.float32)
        result.update(
            {
                "targets": targets,
                "average_residuals": average_residuals,
                "mse": mse(targets, average_predictions),
                "pearson": pearson_corr(targets, average_predictions),
                "mean_residual": (
                    float(np.mean(average_residuals, dtype=np.float64))
                    if average_residuals.size
                    else float("nan")
                ),
            }
        )

    if out_csv is not None:
        out_path = Path(out_csv)
        log_progress(f"ensemble: writing predictions to {out_path}", enabled=progress)
        _fold_ensemble_to_csv(
            out_path,
            ids=result["ids"],
            average_predictions=average_predictions,
            targets=targets,
            average_residuals=average_residuals,
            indices=result["indices"],
        )
        summary: dict[str, object] = {
            "analysis": "fold_checkpoint_ensemble",
            "dataset": str(dataset_path),
            "fold_count": len(paths),
            "checkpoint_paths": [str(path) for path in paths],
            "n_examples": int(indices.shape[0]),
            "target_available": targets is not None,
            "residual_definition": "mean(truth - fold_prediction) = truth - average_prediction",
            "output_csv": str(out_path),
        }
        if targets is not None:
            summary.update(
                {
                    "mse": result["mse"],
                    "pearson": result["pearson"],
                    "mean_residual": result["mean_residual"],
                }
            )
        summary_path = out_path.with_suffix(".summary.json")
        log_progress(f"ensemble: writing summary to {summary_path}", enabled=progress)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_progress("ensemble: done", enabled=progress)
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    out_csv: str | Path | None = None,
    *,
    split: str | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> dict[str, object]:
    """Load a checkpoint and evaluate it on a dataset bundle.

    Args:
        checkpoint_path: TranscriptML checkpoint path to load.
        dataset_path: Processed dataset bundle directory.
        out_csv: Optional destination CSV path for predictions.
        split: Optional named split from the dataset bundle to evaluate.
        batch_size: Number of examples to score per prediction batch.
        device: Torch device used for model execution.
        progress: Whether to emit progress messages while evaluating.
    """

    device = resolve_device(device)
    log_progress(f"evaluate: loading checkpoint {checkpoint_path}", enabled=progress)
    model, _ = load_checkpoint(checkpoint_path, map_location=device)
    log_progress(f"evaluate: loading dataset {dataset_path}", enabled=progress)
    bundle = load_bundle(dataset_path, mmap_mode="r")
    indices = None
    if split is not None:
        if not bundle.splits or split not in bundle.splits:
            raise ValueError(f"Dataset has no split '{split}'")
        indices = [int(i) for i in bundle.splits[split]]
    log_progress(
        f"evaluate: running on {len(indices) if indices is not None else bundle.X.shape[0]} examples",
        enabled=progress,
    )
    result = evaluate_model(model, bundle, indices=indices, batch_size=batch_size, device=device, progress=progress)
    if out_csv is not None:
        log_progress(f"evaluate: writing predictions to {out_csv}", enabled=progress)
        idx = result["indices"]
        ids = [bundle.ids[int(i)] for i in idx]
        targets = result.get("targets")
        predict_to_csv(out_csv, ids=ids, predictions=result["predictions"], targets=targets, indices=idx)
    log_progress("evaluate: done", enabled=progress)
    return result
