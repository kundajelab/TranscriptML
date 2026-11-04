"""Shared optimizer and learning-rate scheduler builders for TranscriptML trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd")
SUPPORTED_LR_SCHEDULERS = (
    "none",
    "reduce_on_plateau",
    "cosine",
    "warmup_cosine",
    "one_cycle",
    "step",
    "exponential",
    "linear",
)


def _config_dict(value: str | Mapping[str, Any] | None, default_name: str) -> dict[str, Any]:
    """Normalize a string, mapping, or ``None`` into ``{"name": ..., "params": {...}}``."""

    if value is None:
        return {"name": default_name, "params": {}}
    if isinstance(value, str):
        return {"name": value.strip().lower(), "params": {}}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a string or mapping for config, got {type(value).__name__}")
    result = dict(value)
    raw_params = result.pop("params", None)
    if raw_params is None:
        params: dict[str, Any] = {}
    elif isinstance(raw_params, Mapping):
        params = dict(raw_params)
    else:
        raise TypeError("config 'params' must be a mapping")
    params.update(result)
    name = str(params.pop("name", default_name)).strip().lower()
    return {"name": name, "params": params}


def build_optimizer(
    model: torch.nn.Module,
    cfg: Any,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Instantiate a PyTorch optimizer from training configuration ``cfg``."""

    config = _config_dict(getattr(cfg, "optimizer", None), "adamw")
    name = config["name"]
    params = dict(config["params"])
    params.setdefault("lr", float(cfg.learning_rate))
    params.setdefault("weight_decay", float(cfg.weight_decay))
    if name == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), **params)
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **params)
    else:
        raise ValueError(f"optimizer must be one of: {', '.join(SUPPORTED_OPTIMIZERS)}")
    return optimizer, {"name": name, "params": params}


@dataclass
class SchedulerBundle:
    """Container managing a PyTorch learning-rate scheduler and its stepping cadence."""

    scheduler: Any | None
    config: dict[str, Any] | None
    step_mode: str = "none"
    monitor: str = "val_loss"

    @property
    def enabled(self) -> bool:
        """Return whether an active learning-rate scheduler is attached."""

        return self.scheduler is not None

    def step_batch(self) -> None:
        """Advance batch-stepped schedulers (such as ``OneCycleLR``) after an optimizer step."""

        if self.scheduler is None or self.step_mode != "batch":
            return
        total_steps = getattr(self.scheduler, "total_steps", None)
        step_count = getattr(self.scheduler, "_step_count", None)
        if total_steps is not None and step_count is not None and int(step_count) > int(total_steps):
            return
        self.scheduler.step()

    def step_epoch(self, metrics: Mapping[str, Any] | None = None) -> None:
        """Advance epoch-stepped or plateau schedulers at the end of an epoch."""

        if self.scheduler is None:
            return
        if self.step_mode == "epoch":
            self.scheduler.step()
            return
        if self.step_mode == "plateau":
            metric = float("nan")
            if metrics is not None:
                raw = metrics.get(self.monitor)
                if raw is not None:
                    metric = float(raw)
                if not np.isfinite(metric):
                    fallback_key = (
                        "train_pearson"
                        if self.monitor.endswith("pearson")
                        else "train_loss"
                    )
                    fallback = metrics.get(fallback_key, metrics.get("loss"))
                    if fallback is not None:
                        metric = float(fallback)
            if np.isfinite(metric):
                self.scheduler.step(metric)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    value: str | Mapping[str, Any] | None,
    *,
    epochs: int,
    steps_per_epoch: int | None = None,
) -> SchedulerBundle:
    """Instantiate a learning-rate scheduler and its normalized configuration."""

    if value is None:
        return SchedulerBundle(scheduler=None, config=None, step_mode="none")

    config = _config_dict(value, "none")
    name = config["name"]
    params = dict(config["params"])
    total_epochs = max(int(epochs), 1)

    if name in {"none", "off", "disabled"}:
        return SchedulerBundle(
            scheduler=None,
            config={"name": "none", "params": {}},
            step_mode="none",
        )

    if name in {"reduce_on_plateau", "plateau"}:
        monitor = str(params.pop("monitor", "val_loss")).strip() or "val_loss"
        default_mode = "max" if monitor.endswith("pearson") else "min"
        params.setdefault("mode", default_mode)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        recorded_params = dict(params)
        if monitor != "val_loss":
            recorded_params["monitor"] = monitor
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "reduce_on_plateau", "params": recorded_params},
            step_mode="plateau",
            monitor=monitor,
        )

    if name in {"cosine", "cosine_annealing"}:
        if "min_lr" in params and "eta_min" not in params:
            params["eta_min"] = float(params.pop("min_lr"))
        params.setdefault("T_max", total_epochs)
        params["T_max"] = max(int(params["T_max"]), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "cosine", "params": params},
            step_mode="epoch",
        )

    if name in {"warmup_cosine", "cosine_with_warmup"}:
        allowed = {"warmup_epochs", "start_factor", "eta_min", "min_lr", "T_max"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown warmup_cosine parameter(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        default_warmup = min(5, max(total_epochs - 1, 0))
        warmup_epochs = int(params.get("warmup_epochs", default_warmup))
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        start_factor = float(params.get("start_factor", 0.1))
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("start_factor must be in (0, 1]")
        eta_min = float(params.get("eta_min", params.get("min_lr", 0.0)))
        t_max = max(int(params.get("T_max", total_epochs)), 1)

        if warmup_epochs == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        elif warmup_epochs >= t_max:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=max(warmup_epochs, 1),
            )
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(t_max - warmup_epochs, 1),
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        normalized_params = {
            "warmup_epochs": warmup_epochs,
            "start_factor": start_factor,
            "eta_min": eta_min,
            "T_max": t_max,
        }
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "warmup_cosine", "params": normalized_params},
            step_mode="epoch",
        )

    if name in {"one_cycle", "onecycle"}:
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        default_max_lr: float | list[float] = (
            base_lrs[0] if len(base_lrs) == 1 else base_lrs
        )
        params.setdefault("max_lr", default_max_lr)
        params.setdefault("pct_start", 0.3)
        params.setdefault("anneal_strategy", "cos")
        params.setdefault("div_factor", 25.0)
        params.setdefault("final_div_factor", 1e4)

        call_params = dict(params)
        if "total_steps" in call_params:
            call_params["total_steps"] = max(int(call_params["total_steps"]), 1)
            step_mode = "batch"
        elif steps_per_epoch is not None and int(steps_per_epoch) > 0:
            call_params.setdefault("epochs", total_epochs)
            call_params.setdefault("steps_per_epoch", int(steps_per_epoch))
            step_mode = "batch"
        else:
            call_params["total_steps"] = total_epochs
            step_mode = "epoch"

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **call_params)
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "one_cycle", "params": call_params},
            step_mode=step_mode,
        )

    if name in {"step", "step_lr"}:
        params.setdefault("step_size", 10)
        params.setdefault("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "step", "params": params},
            step_mode="epoch",
        )

    if name in {"exponential", "exponential_lr"}:
        params.setdefault("gamma", 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "exponential", "params": params},
            step_mode="epoch",
        )

    if name in {"linear", "linear_lr"}:
        params.setdefault("start_factor", 1.0)
        params.setdefault("end_factor", 0.01)
        params.setdefault("total_iters", total_epochs)
        params["total_iters"] = max(int(params["total_iters"]), 1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **params)
        return SchedulerBundle(
            scheduler=scheduler,
            config={"name": "linear", "params": params},
            step_mode="epoch",
        )

    raise ValueError(
        f"lr_scheduler must be one of: {', '.join(SUPPORTED_LR_SCHEDULERS)}"
    )
