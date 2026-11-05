from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import torch
from torch import nn

from transcriptml.models.cnn import SmallCNN, SmallCNNConfig
from transcriptml.models.legnet import LegNet, LegNetConfig
from transcriptml.models.rbpnet import RBPNet, RBPNetConfig
from transcriptml.models.reproduce import SalukiExact, SalukiExactConfig
from transcriptml.models.saluki import SalukiLike, SalukiLikeConfig
from transcriptml.models.saluki_cnn import SalukiCNNHybrid, SalukiCNNHybridConfig
from transcriptml.models.universe import ConvolutionUniverse, ConvolutionUniverseConfig


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
    "rbpnet": ModelSpec(RBPNet, RBPNetConfig),
    "saluki_exact": ModelSpec(SalukiExact, SalukiExactConfig),
    "saluki_cnn_hybrid": ModelSpec(SalukiCNNHybrid, SalukiCNNHybridConfig),
    "convolution_universe": ModelSpec(ConvolutionUniverse, ConvolutionUniverseConfig),
}


def resolve_model_name(name: str) -> str:
    """Resolve a registered model name or model/config class name to its registry key.

    Args:
        name: Registered model name (e.g. ``"saluki_exact"``) or model/config
            class name (e.g. ``"SalukiExact"`` or ``"SalukiExactConfig"``).
    """

    key = str(name).strip()
    if key in MODEL_REGISTRY:
        return key
    lower = key.lower()
    if lower in MODEL_REGISTRY:
        return lower
    for reg_name, spec in MODEL_REGISTRY.items():
        if key in {spec.ctor.__name__, spec.config_cls.__name__} or lower in {
            spec.ctor.__name__.lower(),
            spec.config_cls.__name__.lower(),
        }:
            return reg_name
    raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")


def _merge_param_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested parameter dictionaries."""

    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_param_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def list_models() -> dict[str, str]:
    """Return registered model names mapped to their config class names."""

    return {name: spec.config_cls.__name__ for name, spec in sorted(MODEL_REGISTRY.items())}


def model_default_params(name: str) -> dict[str, Any]:
    """Return default constructor parameters for a registered model.

    Args:
        name: Registered model name or model/config class name.
    """

    resolved_name = resolve_model_name(name)
    spec = MODEL_REGISTRY[resolved_name]
    defaults = spec.config_cls()
    return defaults.to_kwargs() if hasattr(defaults, "to_kwargs") else asdict(defaults)


def normalize_model_config(config: ModelConfig | Mapping[str, Any] | str) -> ModelConfig:
    """Normalize supported model config forms to ``ModelConfig``.

    Args:
        config: ModelConfig instance, mapping with ``name``/``model`` and
            optional ``params``/``kwargs``, or registered model name string.
    """

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
    """Instantiate a registered model from configuration.

    Args:
        config: ModelConfig instance, mapping, or registered model name string.
    """

    cfg = normalize_model_config(config)
    resolved_name = resolve_model_name(cfg.name)
    spec = MODEL_REGISTRY[resolved_name]
    defaults = spec.config_cls()
    default_params = defaults.to_kwargs() if hasattr(defaults, "to_kwargs") else asdict(defaults)
    params = _merge_param_dicts(default_params, cfg.params or {})
    return spec.ctor(**params)


def count_model_parameters(config: ModelConfig | Mapping[str, Any] | str) -> dict[str, Any]:
    """Instantiate a registered model and return parameter count metadata.

    Args:
        config: ModelConfig instance, mapping, or registered model/class name.
    """

    cfg = normalize_model_config(config)
    resolved_name = resolve_model_name(cfg.name)
    spec = MODEL_REGISTRY[resolved_name]
    defaults = spec.config_cls()
    default_params = defaults.to_kwargs() if hasattr(defaults, "to_kwargs") else asdict(defaults)
    resolved_params = _merge_param_dicts(default_params, cfg.params or {})
    model = spec.ctor(**resolved_params)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model": resolved_name,
        "model_class": spec.ctor.__name__,
        "params": resolved_params,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(total_params - trainable_params),
    }


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove a leading DataParallel ``module.`` prefix when present.

    Args:
        state_dict: Model state dictionary whose keys may begin with
            ``"module."``.
    """

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
    """Save model weights, model config, metrics, and optional training state.

    Args:
        path: Destination checkpoint path.
        model: PyTorch module whose state dictionary should be saved.
        model_config: Model configuration used to reconstruct ``model``.
        epoch: Optional epoch number associated with the checkpoint.
        metrics: Optional metrics mapping to include in the checkpoint.
        optimizer_state: Optional optimizer state dictionary to include.
        extra: Optional additional top-level checkpoint fields.
    """

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
    """Load a PyTorch object while supporting older torch versions.

    Args:
        path: Checkpoint or torch object path to load.
        map_location: Torch map-location argument for device remapping.
    """

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
    """Load a TranscriptML checkpoint and rebuild its model.

    Args:
        path: TranscriptML checkpoint path.
        map_location: Torch map-location argument for device remapping.
        strict: Whether to require an exact state-dictionary key match.
    """

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
