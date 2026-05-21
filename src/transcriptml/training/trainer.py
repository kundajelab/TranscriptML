from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from transcriptml.data.bundle import DatasetBundle, load_bundle
from transcriptml.models.common import squeeze_prediction
from transcriptml.models.registry import build_model, normalize_model_config, save_checkpoint
from transcriptml.training.evaluation import evaluate_model, predict_to_csv
from transcriptml.training.metrics import mse, pearson_corr
from transcriptml.training.splits import normalize_splits, predefined_split_indices, random_split_indices


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    model: Mapping[str, Any] = field(default_factory=lambda: {"name": "small_cnn", "params": {}})
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5
    monitor: str = "val_loss"
    device: str = "cpu"
    num_workers: int = 0
    mmap_mode: str | None = "r"
    seed: int = 123
    split: Mapping[str, Any] = field(
        default_factory=lambda: {"method": "random", "val_frac": 0.1, "test_frac": 0.1}
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        return tomllib.loads(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


def _as_train_config(config: TrainConfig | Mapping[str, Any]) -> TrainConfig:
    if isinstance(config, TrainConfig):
        return config
    return TrainConfig(**dict(config))


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_splits(bundle: DatasetBundle, cfg: TrainConfig) -> dict[str, list[int]]:
    if bundle.splits:
        return normalize_splits(bundle.splits)
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


class _ArrayRegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.float32]:
        return np.asarray(self.X[int(idx)]), np.float32(self.y[int(idx)])


def _collate_regression(batch: list[tuple[np.ndarray, np.float32]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.as_tensor(np.stack(xs, axis=0)), torch.as_tensor(np.asarray(ys, dtype=np.float32))


def _loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader | None:
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
    )


def _run_loader(
    model: nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    if loader is None:
        return {"loss": float("nan"), "pearson": float("nan")}
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float().reshape(-1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            yhat = squeeze_prediction(model(xb)).reshape(-1)
            loss = loss_fn(yhat, yb)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()) * int(yb.numel()))
        preds.append(yhat.detach().cpu().numpy())
        targets.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(targets) if targets else np.array([])
    return {
        "loss": float(np.sum(losses) / max(1, y_true.size)),
        "pearson": pearson_corr(y_true, y_pred),
    }


def _is_better(value: float, best: float | None, monitor: str) -> bool:
    if np.isnan(value):
        return False
    if best is None:
        return True
    if monitor.endswith("loss") or monitor in {"loss", "mse", "val_mse"}:
        return value < best
    return value > best


def train_model(bundle: DatasetBundle, config: TrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    cfg = _as_train_config(config)
    if bundle.y is None:
        raise ValueError("Training requires bundle.y")
    _seed_everything(cfg.seed)
    device = _select_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = _make_splits(bundle, cfg)
    model_config = normalize_model_config(cfg.model)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    dataset = _ArrayRegressionDataset(bundle.X, bundle.y)
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        dataset,
        splits["train"],
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
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
    best_value: float | None = None
    best_epoch = -1
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_loader(model, train_loader, device=device, loss_fn=loss_fn, optimizer=optimizer)
        val_metrics = _run_loader(model, val_loader, device=device, loss_fn=loss_fn, optimizer=None)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pearson": train_metrics["pearson"],
            "val_loss": val_metrics["loss"],
            "val_pearson": val_metrics["pearson"],
        }
        history.append(row)
        monitor_value = float(row.get(cfg.monitor, float("nan")))
        if _is_better(monitor_value, best_value, cfg.monitor):
            best_value = monitor_value
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                out / "best.pt",
                model,
                model_config,
                epoch=epoch,
                metrics=row,
                optimizer_state=optimizer.state_dict(),
                extra={"splits": splits, "train_config": asdict(cfg)},
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
            extra={"splits": splits, "train_config": asdict(cfg)},
        )
        if cfg.patience >= 0 and stale > cfg.patience:
            break
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    test_result = evaluate_model(
        model,
        bundle,
        indices=splits.get("test", []),
        batch_size=cfg.batch_size,
        device=device,
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
        "best_monitor": best_value,
        "epochs_run": len(history),
        "test_loss": test_result.get("loss"),
        "test_pearson": test_result.get("pearson"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"model": model, "history": history, "splits": splits, "summary": summary}


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = TrainConfig(**_load_config(config_path))
    bundle = load_bundle(cfg.dataset, mmap_mode=cfg.mmap_mode)
    return train_model(bundle, cfg)
