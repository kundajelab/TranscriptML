from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from transcriptml.data.bundle import DatasetBundle, load_bundle
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
    """Predict scalar outputs for every example in an array."""

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
    """Predict scalar outputs for selected array indices."""

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
    """Evaluate a model on a dataset bundle and optional subset indices."""

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
    """Write prediction rows, and optional targets, to a CSV file."""

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
    """Load a checkpoint and evaluate it on a dataset bundle."""

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
