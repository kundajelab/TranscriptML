from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LossOutput:
    """Scalar loss plus terms used for split-level aggregation."""

    loss: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor


class TrainingLoss(nn.Module):
    """Base class for training losses that may consume auxiliary arrays."""

    name = "loss"
    requires_target = True

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        aux: Mapping[str, torch.Tensor],
    ) -> LossOutput:
        raise NotImplementedError


class RegressionMSELoss(TrainingLoss):
    """Unweighted mean squared error, matching the historical training loss."""

    name = "mse"

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        aux: Mapping[str, torch.Tensor],
    ) -> LossOutput:
        del aux
        err2 = (y_pred - y_true) ** 2
        numerator = err2.sum()
        denominator = y_true.new_tensor(float(max(1, y_true.numel())))
        return LossOutput(loss=numerator / denominator, numerator=numerator, denominator=denominator)


class WeightedMSELoss(TrainingLoss):
    """Weighted MSE using precomputed per-example weights."""

    name = "weighted_mse"

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        aux: Mapping[str, torch.Tensor],
    ) -> LossOutput:
        weights = aux["weight"].to(dtype=y_pred.dtype)
        err2 = (y_pred - y_true) ** 2
        numerator = (weights * err2).sum()
        denominator = weights.sum().clamp_min(torch.finfo(y_pred.dtype).eps)
        return LossOutput(loss=numerator / denominator, numerator=numerator, denominator=denominator)


class BinomialNLLLoss(TrainingLoss):
    """Per-read binomial negative log likelihood for pulse-labeling counts.

    The model prediction is interpreted as log(kdeg). The likelihood omits the
    binomial coefficient because it is constant with respect to model
    parameters; the optimized objective is therefore binomial cross entropy.
    """

    name = "binomial_nll"
    requires_target = False

    def __init__(
        self,
        *,
        eps: float = 1e-7,
        max_rate_time: float = 80.0,
        log_base: str | float = "e",
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.max_rate_time = float(max_rate_time)
        if self.eps <= 0:
            raise ValueError("binomial_nll eps must be positive")
        if self.max_rate_time <= self.eps:
            raise ValueError("binomial_nll max_rate_time must be greater than eps")
        self.log_base_factor = _log_base_factor(log_base)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        aux: Mapping[str, torch.Tensor],
    ) -> LossOutput:
        del y_true
        total = aux["total_reads"].to(dtype=y_pred.dtype)
        new = aux["new_reads"].to(dtype=y_pred.dtype)
        pulse_hours = aux["pulse_hours"].to(dtype=y_pred.dtype)
        old = total - new

        log_kdeg = y_pred * self.log_base_factor
        log_rate_time = log_kdeg + torch.log(pulse_hours.clamp_min(self.eps))
        rate_time = torch.exp(
            log_rate_time.clamp(
                min=math.log(self.eps),
                max=math.log(self.max_rate_time),
            )
        )

        prob_new = (-torch.expm1(-rate_time)).clamp(min=self.eps, max=1.0)
        log_new = torch.log(prob_new)
        log_old = -rate_time
        nll = -(new * log_new + old * log_old)
        numerator = nll.sum()
        denominator = total.sum().clamp_min(1.0)
        return LossOutput(loss=numerator / denominator, numerator=numerator, denominator=denominator)


def build_training_loss(
    config: str | Mapping[str, Any] | None,
    *,
    metadata: Sequence[Mapping[str, Any]] | None,
    n_examples: int,
) -> tuple[TrainingLoss, dict[str, np.ndarray], dict[str, Any]]:
    """Build a configured training loss and aligned auxiliary arrays.

    Args:
        config: Loss configuration. ``None`` and ``"mse"`` preserve historical
            unweighted MSE behavior.
        metadata: Optional bundle metadata aligned to examples.
        n_examples: Number of examples in the bundle.
    """

    cfg = _normalize_loss_config(config)
    name = str(cfg.get("name", "mse")).strip().lower()
    if name in {"mse", "mean_squared_error"}:
        return RegressionMSELoss(), {}, {"name": "mse"}
    if name in {"weighted_mse", "weighted_mean_squared_error"}:
        aux, normalized = _weighted_mse_aux(cfg, metadata=metadata, n_examples=n_examples)
        return WeightedMSELoss(), aux, normalized
    if name in {"binomial_nll", "binomial", "binomial_count", "binomial_count_nll"}:
        aux, normalized = _binomial_aux(cfg, metadata=metadata, n_examples=n_examples)
        return BinomialNLLLoss(
            eps=float(normalized["eps"]),
            max_rate_time=float(normalized["max_rate_time"]),
            log_base=normalized["log_base"],
        ), aux, normalized
    raise ValueError(f"Unknown training loss '{name}'")


def _normalize_loss_config(config: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {"name": "mse"}
    if isinstance(config, str):
        return {"name": config}
    return dict(config)


def _weighted_mse_aux(
    cfg: Mapping[str, Any],
    *,
    metadata: Sequence[Mapping[str, Any]] | None,
    n_examples: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weight_col = _optional_str(cfg.get("weight_col"))
    se_col = _first_config_str(cfg, "se_col", "sigma_col", "log_kdeg_se_col")
    if bool(weight_col) == bool(se_col):
        raise ValueError("weighted_mse requires exactly one of weight_col or se_col")

    min_weight = _optional_float(cfg.get("min_weight", 0.01))
    max_weight = _optional_float(cfg.get("max_weight", 100.0))
    if min_weight is not None and min_weight < 0:
        raise ValueError("weighted_mse min_weight must be nonnegative")
    if max_weight is not None and max_weight <= 0:
        raise ValueError("weighted_mse max_weight must be positive")
    if min_weight is not None and max_weight is not None and min_weight > max_weight:
        raise ValueError("weighted_mse min_weight must be <= max_weight")

    if weight_col:
        weights = _metadata_float_array(metadata, weight_col, n_examples=n_examples)
        source = "weight_col"
        source_col = weight_col
    else:
        eps = float(cfg.get("eps", 1e-8))
        if eps <= 0:
            raise ValueError("weighted_mse eps must be positive")
        se = _metadata_float_array(metadata, se_col or "", n_examples=n_examples)
        if np.any(se < 0):
            raise ValueError(f"weighted_mse se_col '{se_col}' contains negative values")
        weights = 1.0 / (se * se + eps)
        source = "se_col"
        source_col = se_col

    if np.any(weights < 0):
        raise ValueError(f"weighted_mse {source} '{source_col}' contains negative weights")
    if min_weight is not None or max_weight is not None:
        weights = np.clip(
            weights,
            a_min=-np.inf if min_weight is None else min_weight,
            a_max=np.inf if max_weight is None else max_weight,
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"weighted_mse {source} '{source_col}' produced non-finite weights")
    if float(np.sum(weights)) <= 0:
        raise ValueError("weighted_mse weights must sum to a positive value")

    normalized: dict[str, Any] = {
        "name": "weighted_mse",
        source: source_col,
        "min_weight": min_weight,
        "max_weight": max_weight,
    }
    if not weight_col:
        normalized["eps"] = float(cfg.get("eps", 1e-8))
    return {"weight": weights.astype(np.float32)}, normalized


def _binomial_aux(
    cfg: Mapping[str, Any],
    *,
    metadata: Sequence[Mapping[str, Any]] | None,
    n_examples: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    total_col = _optional_str(cfg.get("total_reads_col", "total_reads")) or "total_reads"
    new_col = _optional_str(cfg.get("new_reads_col", "new_reads")) or "new_reads"
    pulse_col = _optional_str(cfg.get("pulse_hours_col", "pulse_hours")) or "pulse_hours"
    total = _metadata_float_array(metadata, total_col, n_examples=n_examples)
    new = _metadata_float_array(metadata, new_col, n_examples=n_examples)
    pulse_hours = _metadata_float_array(metadata, pulse_col, n_examples=n_examples)

    if np.any(total <= 0):
        raise ValueError(f"binomial_nll total_reads_col '{total_col}' must be positive")
    if np.any(new < 0):
        raise ValueError(f"binomial_nll new_reads_col '{new_col}' contains negative values")
    if np.any(new > total):
        raise ValueError(f"binomial_nll new_reads_col '{new_col}' exceeds total_reads_col '{total_col}'")
    if np.any(pulse_hours <= 0):
        raise ValueError(f"binomial_nll pulse_hours_col '{pulse_col}' must be positive")

    eps = float(cfg.get("eps", 1e-7))
    max_rate_time = float(cfg.get("max_rate_time", 80.0))
    log_base = cfg.get("log_base", "e")
    _log_base_factor(log_base)
    normalized = {
        "name": "binomial_nll",
        "total_reads_col": total_col,
        "new_reads_col": new_col,
        "pulse_hours_col": pulse_col,
        "eps": eps,
        "max_rate_time": max_rate_time,
        "log_base": log_base,
    }
    aux = {
        "total_reads": total.astype(np.float32),
        "new_reads": new.astype(np.float32),
        "pulse_hours": pulse_hours.astype(np.float32),
    }
    return aux, normalized


def _metadata_float_array(
    metadata: Sequence[Mapping[str, Any]] | None,
    col: str,
    *,
    n_examples: int,
) -> np.ndarray:
    if metadata is None:
        raise ValueError(f"Loss requires metadata column '{col}', but bundle.metadata is missing")
    if len(metadata) != n_examples:
        raise ValueError("bundle.metadata length must match X.shape[0] for metadata-aware losses")
    values: list[float] = []
    missing: list[int] = []
    bad: list[int] = []
    for i, row in enumerate(metadata):
        if col not in row or row[col] is None or str(row[col]).strip() == "":
            missing.append(i)
            continue
        try:
            values.append(float(row[col]))
        except (TypeError, ValueError):
            bad.append(i)
            values.append(float("nan"))
    if missing:
        raise ValueError(f"Missing metadata column '{col}' for {len(missing)} example(s); first index {missing[0]}")
    if bad:
        raise ValueError(f"Metadata column '{col}' has non-numeric values; first index {bad[0]}")
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] != n_examples:
        raise ValueError(f"Metadata column '{col}' did not produce {n_examples} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Metadata column '{col}' contains non-finite values")
    return arr


def _first_config_str(cfg: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _optional_str(cfg.get(key))
        if value:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _log_base_factor(log_base: str | float) -> float:
    if isinstance(log_base, str):
        text = log_base.strip().lower()
        if text in {"e", "natural", "ln"}:
            return 1.0
        if text in {"10", "log10"}:
            return math.log(10.0)
        if text in {"2", "log2"}:
            return math.log(2.0)
        try:
            base = float(text)
        except ValueError as exc:
            raise ValueError("log_base must be 'e', '10', '2', or a positive numeric base") from exc
    else:
        base = float(log_base)
    if base <= 0 or base == 1:
        raise ValueError("log_base must be positive and not equal to 1")
    return math.log(base)
