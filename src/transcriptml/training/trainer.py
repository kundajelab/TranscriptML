from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from transcriptml.data.bundle import DatasetBundle, load_bundle
from transcriptml.data.controls import apply_sequence_controls_to_bundle
from transcriptml.devices import resolve_device
from transcriptml.models.common import squeeze_prediction
from transcriptml.models.registry import build_model, normalize_model_config, save_checkpoint
from transcriptml.training.evaluation import evaluate_model, predict_to_csv
from transcriptml.training.losses import TrainingLoss, build_training_loss
from transcriptml.training.metrics import pearson_corr
from transcriptml.training.splits import normalize_splits, predefined_split_indices, random_split_indices
from transcriptml.progress import ProgressReporter, log_progress


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    model: Mapping[str, Any] = field(default_factory=lambda: {"name": "small_cnn", "params": {}})
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 0.5
    patience: int = 5
    monitor: str | Sequence[str] = "val_loss"
    loss: str | Mapping[str, Any] | None = field(default_factory=lambda: {"name": "mse"})
    device: str = "cpu"
    num_workers: int = 0
    mmap_mode: str | None = "r"
    seed: int = 123
    progress: bool = True
    sequence_controls: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None
    split_source: str = "auto"
    split: Mapping[str, Any] = field(
        default_factory=lambda: {"method": "random", "val_frac": 0.1, "test_frac": 0.1}
    )


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Integer seed applied across supported random number generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or TOML training configuration file.

    Args:
        path: Path to a ``.json`` or ``.toml`` config file.
    """

    p = Path(path)
    if p.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        return tomllib.loads(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _as_train_config(config: TrainConfig | Mapping[str, Any]) -> TrainConfig:
    """Normalize mapping-like training config to ``TrainConfig``.

    Args:
        config: Existing ``TrainConfig`` or mapping of constructor fields.
    """

    if isinstance(config, TrainConfig):
        return config
    return TrainConfig(**dict(config))


def _make_config_splits(bundle: DatasetBundle, cfg: TrainConfig) -> dict[str, list[int]]:
    """Choose dataset splits from the training config.

    Args:
        bundle: Dataset bundle used for size and metadata.
        cfg: Training configuration containing split strategy settings.
    """

    split_cfg = dict(cfg.split or {})
    method = split_cfg.get("method", "random")
    if method == "random":
        return random_split_indices(
            bundle.X.shape[0],
            val_frac=float(split_cfg.get("val_frac", 0.1)),
            test_frac=float(split_cfg.get("test_frac", 0.1)),
            seed=int(split_cfg.get("seed", cfg.seed)),
        )
    if method == "metadata":
        if bundle.metadata is None:
            raise ValueError("metadata split requested but bundle has no metadata")
        return predefined_split_indices(bundle.metadata, split_col=str(split_cfg.get("split_col", "split")))
    if method == "predefined":
        return normalize_splits(split_cfg["splits"])
    raise ValueError(f"Unknown split method '{method}'")


def _select_splits(bundle: DatasetBundle, cfg: TrainConfig) -> tuple[dict[str, list[int]], str]:
    """Choose dataset splits and report which source was used.

    Args:
        bundle: Dataset bundle that may already contain predefined splits.
        cfg: Training configuration containing split source and strategy.
    """

    source = str(cfg.split_source or "auto").strip().lower()
    if source not in {"auto", "bundle", "config"}:
        raise ValueError("split_source must be one of: auto, bundle, config")
    has_bundle_splits = bundle.splits is not None
    if source == "bundle":
        if not has_bundle_splits:
            raise ValueError("split_source='bundle' requested but dataset bundle has no splits")
        return normalize_splits(bundle.splits), "bundle"
    if source == "config":
        return _make_config_splits(bundle, cfg), "config"
    if has_bundle_splits:
        return normalize_splits(bundle.splits), "bundle"
    return _make_config_splits(bundle, cfg), "config"


def _make_splits(bundle: DatasetBundle, cfg: TrainConfig) -> dict[str, list[int]]:
    """Choose dataset splits from the bundle or training config."""

    splits, _ = _select_splits(bundle, cfg)
    return splits


class _ArrayRegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aux_arrays: Mapping[str, np.ndarray] | None = None):
        """Wrap NumPy arrays as a PyTorch regression dataset.

        Args:
            X: Encoded input array with examples on axis 0.
            y: Scalar target array aligned to ``X``.
            aux_arrays: Optional per-example auxiliary arrays used by
                metadata-aware losses.
        """

        self.X = X
        self.y = y
        self.aux_arrays = dict(aux_arrays or {})

    def __len__(self) -> int:
        """Return the number of examples."""

        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32, dict[str, np.float32]]:
        """Return one input array and scalar target.

        Args:
            idx: Integer example index to retrieve.
        """

        i = int(idx)
        aux = {name: np.float32(values[i]) for name, values in self.aux_arrays.items()}
        return np.asarray(self.X[i]), np.float32(self.y[i]), aux


def _collate_regression(
    batch: list[tuple[np.ndarray, np.float32, Mapping[str, np.float32]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Stack NumPy regression examples into tensors.

    Args:
        batch: List of ``(input_array, scalar_target, aux)`` examples from the
            dataset.
    """

    xs, ys, auxs = zip(*batch)
    aux_keys = auxs[0].keys() if auxs else []
    aux = {
        key: torch.as_tensor(np.asarray([item[key] for item in auxs], dtype=np.float32))
        for key in aux_keys
    }
    return torch.as_tensor(np.stack(xs, axis=0)), torch.as_tensor(np.asarray(ys, dtype=np.float32)), aux


def _loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader | None:
    """Create a DataLoader for a split, or ``None`` for empty splits.

    Args:
        dataset: PyTorch dataset containing all examples.
        indices: Example indices assigned to the split.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the split each epoch.
        num_workers: Number of worker processes used by the DataLoader.
        pin_memory: Whether the DataLoader should pin host memory.
        drop_last: Whether to drop the final incomplete batch.
    """

    if not indices:
        return None
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(num_workers) > 0,
        collate_fn=_collate_regression,
        drop_last=drop_last,
    )


def _would_create_singleton_batch(n_examples: int, batch_size: int) -> bool:
    """Return whether batching would leave one training example by itself."""

    n = int(n_examples)
    bs = int(batch_size)
    if bs <= 1 or n <= bs:
        return False
    return n % bs == 1


def _run_loader(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    loss_fn: TrainingLoss,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    target_metrics: bool = True,
    progress: bool = True,
    progress_label: str | None = None,
) -> dict[str, float]:
    """Run one train or evaluation pass over a loader.

    Args:
        model: PyTorch model to train or evaluate.
        loader: DataLoader for a split, or ``None`` for an empty split.
        device: Torch device used for tensors and model execution.
        loss_fn: Loss module used to compare predictions and targets.
        optimizer: Optional optimizer. When provided, gradients are updated.
        gradient_clip_norm: Optional positive norm for gradient clipping during
            training.
        target_metrics: Whether to compute target-based metrics such as
            Pearson correlation.
        progress: Whether to emit progress messages while iterating.
        progress_label: Optional label shown in progress messages.
    """

    if loader is None:
        return {"loss": float("nan"), "pearson": float("nan")}
    training = optimizer is not None
    model.train(training)
    loss_numerator = 0.0
    loss_denominator = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    reporter = ProgressReporter(
        progress_label or ("train batches" if training else "eval batches"),
        total=len(loader),
        unit="batches",
        enabled=progress,
        percent_step=25.0,
    )
    for xb, yb, auxb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float().reshape(-1)
        auxb = {name: values.to(device).float().reshape(-1) for name, values in auxb.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            yhat = squeeze_prediction(model(xb)).reshape(-1)
            loss_output = loss_fn(yhat, yb, auxb)
            loss = loss_output.loss
            if training:
                loss.backward()
                if gradient_clip_norm is not None and float(gradient_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
                optimizer.step()
        loss_numerator += float(loss_output.numerator.detach().cpu().item())
        loss_denominator += float(loss_output.denominator.detach().cpu().item())
        preds.append(yhat.detach().cpu().numpy())
        if target_metrics:
            targets.append(yb.detach().cpu().numpy())
        reporter.update()
    reporter.close()
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(targets) if targets else np.array([])
    return {
        "loss": float(loss_numerator / max(loss_denominator, 1e-12)),
        "pearson": pearson_corr(y_true, y_pred) if target_metrics else float("nan"),
    }


def _is_better(value: float, best: float | None, monitor: str) -> bool:
    """Return whether a monitored metric improved.

    Args:
        value: Current metric value.
        best: Previous best metric value, or ``None`` if unset.
        monitor: Metric name used to decide whether lower or higher is better.
    """

    if np.isnan(value):
        return False
    if best is None:
        return True
    if monitor.endswith("loss") or monitor in {"loss", "mse", "val_mse"}:
        return value < best
    return value > best


def _monitor_names(monitor: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one or more monitored metric names.

    Args:
        monitor: Comma-separated metric string or sequence of metric names.
    """

    if isinstance(monitor, str):
        names = [part.strip() for part in monitor.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in monitor if str(part).strip()]
    if not names:
        raise ValueError("monitor must name at least one metric")
    return tuple(names)


def _format_best_metrics(best_metrics: Mapping[str, float | None]) -> str:
    """Format monitored best values for progress output.

    Args:
        best_metrics: Mapping from monitor names to their current best values.
    """

    return ", ".join(f"best_{name}={value}" for name, value in best_metrics.items())


def _monitor_improved(
    row: Mapping[str, float | int],
    monitors: Sequence[str],
    best_metrics: Mapping[str, float | None],
) -> tuple[bool, dict[str, float]]:
    """Return whether any monitored metric improved over the current best epoch.

    Args:
        row: Current epoch metrics keyed by metric name.
        monitors: Metric names to compare against ``best_metrics``.
        best_metrics: Previous best values for each monitored metric.
    """

    missing_monitors = [name for name in monitors if name not in row]
    if missing_monitors:
        raise ValueError(f"Unknown monitor metric(s): {', '.join(missing_monitors)}")
    values = {name: float(row[name]) for name in monitors}
    improved = any(_is_better(values[name], best_metrics[name], name) for name in monitors)
    return improved, values


def train_model(bundle: DatasetBundle, config: TrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Train a model from an in-memory dataset bundle and config.

    Args:
        bundle: Dataset bundle containing encoded inputs and regression targets.
        config: Training configuration object or mapping of config fields.
    """

    cfg = _as_train_config(config)
    _seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sequence_control_stats: dict[str, Any] | None = None
    if cfg.sequence_controls:
        bundle, sequence_control_stats = apply_sequence_controls_to_bundle(
            bundle,
            cfg.sequence_controls,
            default_save_dir=out / "sequence_controlled_dataset",
            progress=cfg.progress,
        )
    loss_fn, aux_arrays, normalized_loss_config = build_training_loss(
        cfg.loss,
        metadata=bundle.metadata,
        n_examples=int(bundle.X.shape[0]),
    )
    has_targets = bundle.y is not None
    if bundle.y is None:
        if loss_fn.requires_target:
            raise ValueError(f"Training loss '{loss_fn.name}' requires bundle.y")
        y_train = np.zeros(int(bundle.X.shape[0]), dtype=np.float32)
    else:
        y_train = bundle.y
    splits, split_source_used = _select_splits(bundle, cfg)
    split_counts = {name: len(splits.get(name, [])) for name in ("train", "val", "test")}
    model_config = normalize_model_config(cfg.model)
    log_progress(
        (
            "training: "
            f"device={device}, output={out}, loss={normalized_loss_config['name']}, "
            f"split_source={split_source_used} requested={cfg.split_source}, "
            f"train={split_counts['train']}, val={split_counts['val']}, "
            f"test={split_counts['test']}"
        ),
        enabled=cfg.progress,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    dataset = _ArrayRegressionDataset(bundle.X, y_train, aux_arrays)
    pin_memory = device.type == "cuda"
    drop_last_train = _would_create_singleton_batch(len(splits["train"]), cfg.batch_size)
    if drop_last_train:
        log_progress(
            (
                "training: train split would end with a singleton batch; "
                "dropping the final shuffled training example each epoch"
            ),
            enabled=cfg.progress,
        )
    train_loader = _loader(
        dataset,
        splits["train"],
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = _loader(
        dataset,
        splits["val"],
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    history: list[dict[str, float | int]] = []
    monitors = _monitor_names(cfg.monitor)
    best_metrics: dict[str, float | None] = {name: None for name in monitors}
    best_epoch = -1
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        log_progress(f"epoch {epoch}/{cfg.epochs}: starting", enabled=cfg.progress)
        train_metrics = _run_loader(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            gradient_clip_norm=cfg.gradient_clip_norm,
            target_metrics=has_targets,
            progress=cfg.progress,
            progress_label=f"epoch {epoch} train",
        )
        val_metrics = _run_loader(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            target_metrics=has_targets,
            progress=cfg.progress,
            progress_label=f"epoch {epoch} val",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pearson": train_metrics["pearson"],
            "val_loss": val_metrics["loss"],
            "val_pearson": val_metrics["pearson"],
        }
        history.append(row)
        improved, monitor_values = _monitor_improved(row, monitors, best_metrics)
        if improved:
            best_metrics = dict(monitor_values)
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                out / "best.pt",
                model,
                model_config,
                epoch=epoch,
                metrics=row,
                optimizer_state=optimizer.state_dict(),
                extra={
                    "splits": splits,
                    "split_source_used": split_source_used,
                    "train_config": asdict(cfg),
                    "loss_config": normalized_loss_config,
                },
            )
        else:
            stale += 1
        save_checkpoint(
            out / "last.pt",
            model,
            model_config,
            epoch=epoch,
            metrics=row,
            optimizer_state=optimizer.state_dict(),
            extra={
                "splits": splits,
                "split_source_used": split_source_used,
                "train_config": asdict(cfg),
                "loss_config": normalized_loss_config,
            },
        )
        log_progress(
            (
                f"epoch {epoch}/{cfg.epochs}: "
                f"train_loss={row['train_loss']:.6g}, train_pearson={row['train_pearson']:.4g}, "
                f"val_loss={row['val_loss']:.6g}, val_pearson={row['val_pearson']:.4g}, "
                f"{_format_best_metrics(best_metrics)}"
            ),
            enabled=cfg.progress,
        )
        if cfg.patience >= 0 and stale > cfg.patience:
            log_progress(f"early stopping after epoch {epoch}: patience exceeded", enabled=cfg.progress)
            break
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # Well there's a chance there is none, so kind of weird...
    # Not that a user should ever not want to include a test split
    log_progress("training: evaluating test split", enabled=cfg.progress)
    test_loader = _loader(
        dataset,
        splits.get("test", []),
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loss_metrics = _run_loader(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=None,
        target_metrics=has_targets,
        progress=cfg.progress,
        progress_label="test loss",
    )
    test_result = evaluate_model(
        model,
        bundle,
        indices=splits.get("test", []),
        batch_size=cfg.batch_size,
        device=device,
        progress=cfg.progress,
    )
    if splits.get("test"):
        ids = [bundle.ids[int(i)] for i in test_result["indices"]]
        predict_to_csv(
            out / "test_predictions.csv",
            ids=ids,
            predictions=test_result["predictions"],
            targets=test_result.get("targets"),
            indices=test_result["indices"],
        )
    summary = {
        "best_epoch": best_epoch,
        "monitor": list(monitors),
        "best_monitor_values": best_metrics,
        "epochs_run": len(history),
        "loss": normalized_loss_config,
        "split_source_requested": cfg.split_source,
        "split_source_used": split_source_used,
        "split_counts": split_counts,
        "test_loss": test_loss_metrics.get("loss"),
        "test_mse": test_result.get("loss"),
        "test_pearson": test_result.get("pearson"),
    }
    if sequence_control_stats is not None:
        summary["sequence_controls"] = sequence_control_stats
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress(
        f"training: done; best_epoch={best_epoch}, summary={out / 'summary.json'}",
        enabled=cfg.progress,
    )

    # Not sure this return value ever gets used anywhere
    return {"model": model, "history": history, "splits": splits, "summary": summary}


def train_from_config(config_path: str | Path, *, progress: bool | None = None) -> dict[str, Any]:
    """Load a training config and train its requested model.

    Args:
        config_path: Path to a JSON or TOML training configuration file.
        progress: Optional override for whether progress messages are emitted.
    """

    cfg = TrainConfig(**_load_config(config_path))
    if progress is not None:
        cfg.progress = bool(progress)
    log_progress(f"training: loading dataset {cfg.dataset}", enabled=cfg.progress)
    bundle = load_bundle(cfg.dataset, mmap_mode=cfg.mmap_mode)
    return train_model(bundle, cfg)
