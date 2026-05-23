from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import torch
from torch import nn

from transcriptml.models.cnn import SmallCNN, SmallCNNConfig
from transcriptml.models.legnet import LegNet, LegNetConfig
from transcriptml.models.reproduce import SalukiExact, SalukiExactConfig
from transcriptml.models.saluki import SalukiLike, SalukiLikeConfig


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize model name and parameters to a plain dictionary."""

        return {"name": self.name, "params": dict(self.params or {})}


@dataclass(frozen=True)
class ModelSpec:
    ctor: Callable[..., nn.Module]
    config_cls: type


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "small_cnn": ModelSpec(SmallCNN, SmallCNNConfig),
    "saluki_like": ModelSpec(SalukiLike, SalukiLikeConfig),
    "saluki_gru": ModelSpec(SalukiLike, SalukiLikeConfig),
    "legnet": ModelSpec(LegNet, LegNetConfig),
    "saluki_exact": ModelSpec(SalukiExact, SalukiExactConfig),
}


def normalize_model_config(config: ModelConfig | Mapping[str, Any] | str) -> ModelConfig:
    """Normalize supported model config forms to ``ModelConfig``."""

    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, str):
        return ModelConfig(name=config, params={})
    name = str(config.get("name", config.get("model", "")))
    if not name:
        raise ValueError("Model config must include 'name'")
    params = config.get("params", config.get("kwargs", {})) or {}
    if is_dataclass(params):
        params = asdict(params)
    return ModelConfig(name=name, params=dict(params))


def build_model(config: ModelConfig | Mapping[str, Any] | str) -> nn.Module:
    """Instantiate a registered model from configuration."""

    cfg = normalize_model_config(config)
    try:
        spec = MODEL_REGISTRY[cfg.name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{cfg.name}'. Available: {sorted(MODEL_REGISTRY)}") from exc
    defaults = spec.config_cls()
    params = defaults.to_kwargs() if hasattr(defaults, "to_kwargs") else asdict(defaults)
    params.update(cfg.params or {})
    return spec.ctor(**params)


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove a leading DataParallel ``module.`` prefix when present."""

    if not state_dict:
        return dict(state_dict)
    first = next(iter(state_dict))
    if first.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return dict(state_dict)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: ModelConfig | Mapping[str, Any],
    *,
    epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save model weights, model config, metrics, and optional training state."""

    cfg = normalize_model_config(model_config).to_dict()
    obj: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": cfg,
        "epoch": epoch,
        "metrics": dict(metrics or {}),
    }
    if optimizer_state is not None:
        obj["optimizer_state"] = optimizer_state
    if extra:
        obj.update(dict(extra))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch object while supporting older torch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a TranscriptML checkpoint and rebuild its model."""

    ckpt = _torch_load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary saved by save_checkpoint")
    if "model_config" not in ckpt:
        raise ValueError("Checkpoint is missing model_config; cannot rebuild model")
    model = build_model(ckpt["model_config"])
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise ValueError("Checkpoint is missing state_dict")
    model.load_state_dict(_strip_module_prefix(state), strict=strict)
    model.eval()
    return model, ckpt
