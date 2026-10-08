from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import torch
from torch import nn

from transcriptml.models.cnn import SmallCNN
from transcriptml.models.common import dropout_or_identity
from transcriptml.models.reproduce import SalukiExact, StochasticShift, keras_he_normal_trunc_


@dataclass
class _SalukiBranchConfig:
    seq_depth: int = 6
    filters: int = 64
    kernel_size: int = 5
    num_layers: int = 6
    pooling: str = "max"
    dropout: float = 0.3
    augment_shift: int = 3
    ln_epsilon: float = 0.007


@dataclass
class _CNNBranchConfig:
    in_ch: int = 6
    n_filters: int = 64
    kernel_size: int = 9
    n_layers: int = 2
    dropout: float = 0.1


@dataclass
class _FusionConfig:
    mode: str = "concat"
    saluki_projection_dim: int = 64
    cnn_projection_dim: int = 64
    projection_norm: str = "layer"
    projection_activation: str = "gelu"
    projection_dropout: float = 0.0
    head_hidden: int = 64
    head_norm: str = "batch"
    head_activation: str = "relu"
    head_dropout: float = 0.3
    head_norm_eps: float = 0.001
    head_bn_momentum: float = 0.1
    output_dim: int = 1
    weight_init: str = "saluki"


def _saluki_defaults() -> dict[str, Any]:
    return asdict(_SalukiBranchConfig())


def _cnn_defaults() -> dict[str, Any]:
    return asdict(_CNNBranchConfig())


def _fusion_defaults() -> dict[str, Any]:
    return asdict(_FusionConfig())


@dataclass
class SalukiCNNHybridConfig:
    """Nested configuration for :class:`SalukiCNNHybrid`."""

    saluki: dict[str, Any] = field(default_factory=_saluki_defaults)
    cnn: dict[str, Any] = field(default_factory=_cnn_defaults)
    fusion: dict[str, Any] = field(default_factory=_fusion_defaults)

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``SalukiCNNHybrid``."""

        return asdict(self)


def _merge_section(value: object, config_cls: type, name: str):
    defaults = asdict(config_cls())
    if value is None:
        supplied: dict[str, Any] = {}
    elif is_dataclass(value):
        supplied = asdict(value)
    elif isinstance(value, Mapping):
        supplied = dict(value)
    else:
        raise TypeError(f"{name} must be a mapping or dataclass")
    allowed = {item.name for item in fields(config_cls)}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} parameter(s): {unknown}")
    defaults.update(supplied)
    return config_cls(**defaults)


def _validate_probability(name: str, value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")
    return probability


def _validate_choice(name: str, value: str, choices: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return normalized


def _activation(kind: str, *, allow_none: bool = False) -> nn.Module:
    choices = {"relu", "gelu", "silu"}
    if allow_none:
        choices.add("none")
    normalized = _validate_choice("activation", kind, choices)
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    return nn.Identity()


def _feature_norm(
    kind: str,
    features: int,
    *,
    eps: float,
    batch_momentum: float,
) -> nn.Module:
    normalized = _validate_choice("normalization", kind, {"none", "batch", "layer"})
    if normalized == "batch":
        return nn.BatchNorm1d(features, eps=float(eps), momentum=float(batch_momentum))
    if normalized == "layer":
        return nn.LayerNorm(features, eps=float(eps))
    return nn.Identity()


class _SalukiFeatureEncoder(nn.Module):
    """SalukiExact through its reversed GRU, without the standalone head."""

    def __init__(self, config: _SalukiBranchConfig):
        super().__init__()
        source = SalukiExact(
            seq_depth=config.seq_depth,
            filters=config.filters,
            kernel_size=config.kernel_size,
            num_layers=config.num_layers,
            pooling=config.pooling,
            dropout=config.dropout,
            augment_shift=0,
            ln_epsilon=config.ln_epsilon,
        )
        self.conv0 = source.conv0
        self.blocks = source.blocks
        self.pre_rnn_ln = source.pre_rnn_ln
        self.pre_rnn_act = source.pre_rnn_act
        self.gru = source.gru
        self.output_dim = int(config.filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv0(x)
        for block in self.blocks:
            z = block["ln"](z)
            z = block["act"](z)
            z = block["conv"](z)
            z = block["drop"](z)
            z = block["pool"](z)
        z = self.pre_rnn_act(self.pre_rnn_ln(z)).transpose(1, 2)
        z = torch.flip(z, dims=[1])
        z, _ = self.gru(z)
        return z[:, -1, :]


class _SmallCNNFeatureEncoder(nn.Module):
    """SmallCNN through its max-plus-mean pooling, without the standalone head."""

    def __init__(self, config: _CNNBranchConfig):
        super().__init__()
        source = SmallCNN(
            in_ch=config.in_ch,
            n_filters=config.n_filters,
            kernel_size=config.kernel_size,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        self.encoder = source.encoder
        self.output_dim = int(config.n_filters) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x.float())
        return torch.cat([z.amax(dim=-1), z.mean(dim=-1)], dim=1)


class _FeatureProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        norm: str,
        activation: str,
        dropout: float,
        norm_eps: float,
        batch_momentum: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = _feature_norm(
            norm,
            out_features,
            eps=norm_eps,
            batch_momentum=batch_momentum,
        )
        self.activation = _activation(activation, allow_none=True)
        self.dropout = dropout_or_identity(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.linear(x))))


class _SalukiPredictionHead(nn.Module):
    """Saluki pre-normalized two-layer prediction head."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        output_dim: int,
        *,
        norm: str,
        activation: str,
        dropout: float,
        norm_eps: float,
        batch_momentum: float,
    ):
        super().__init__()
        self.norm1 = _feature_norm(
            norm,
            in_features,
            eps=norm_eps,
            batch_momentum=batch_momentum,
        )
        self.activation1 = _activation(activation)
        self.linear1 = nn.Linear(in_features, hidden)
        self.dropout = dropout_or_identity(dropout)
        self.norm2 = _feature_norm(
            norm,
            hidden,
            eps=norm_eps,
            batch_momentum=batch_momentum,
        )
        self.activation2 = _activation(activation)
        self.linear2 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear1(self.activation1(self.norm1(x)))
        z = self.dropout(z)
        return self.linear2(self.activation2(self.norm2(z)))


class SalukiCNNHybrid(nn.Module):
    """Hybrid of exact Saluki and SmallCNN feature encoders."""

    def __init__(
        self,
        saluki: Mapping[str, Any] | object | None = None,
        cnn: Mapping[str, Any] | object | None = None,
        fusion: Mapping[str, Any] | object | None = None,
    ):
        super().__init__()
        saluki_config = _merge_section(saluki, _SalukiBranchConfig, "saluki")
        cnn_config = _merge_section(cnn, _CNNBranchConfig, "cnn")
        fusion_config = _merge_section(fusion, _FusionConfig, "fusion")
        self._validate_configs(saluki_config, cnn_config, fusion_config)

        self.in_ch = int(saluki_config.seq_depth)
        self.output_dim = int(fusion_config.output_dim)
        self.fusion_mode = _validate_choice(
            "fusion.mode", fusion_config.mode, {"concat", "project_concat"}
        )
        self.shift = StochasticShift(int(saluki_config.augment_shift))
        self.saluki_encoder = _SalukiFeatureEncoder(saluki_config)
        self.cnn_encoder = _SmallCNNFeatureEncoder(cnn_config)

        if self.fusion_mode == "project_concat":
            self.saluki_projection: nn.Module = _FeatureProjection(
                self.saluki_encoder.output_dim,
                int(fusion_config.saluki_projection_dim),
                norm=fusion_config.projection_norm,
                activation=fusion_config.projection_activation,
                dropout=float(fusion_config.projection_dropout),
                norm_eps=float(fusion_config.head_norm_eps),
                batch_momentum=float(fusion_config.head_bn_momentum),
            )
            self.cnn_projection: nn.Module = _FeatureProjection(
                self.cnn_encoder.output_dim,
                int(fusion_config.cnn_projection_dim),
                norm=fusion_config.projection_norm,
                activation=fusion_config.projection_activation,
                dropout=float(fusion_config.projection_dropout),
                norm_eps=float(fusion_config.head_norm_eps),
                batch_momentum=float(fusion_config.head_bn_momentum),
            )
            fused_dim = int(fusion_config.saluki_projection_dim) + int(
                fusion_config.cnn_projection_dim
            )
        else:
            self.saluki_projection = nn.Identity()
            self.cnn_projection = nn.Identity()
            fused_dim = self.saluki_encoder.output_dim + self.cnn_encoder.output_dim

        self.fused_dim = fused_dim
        self.head = _SalukiPredictionHead(
            fused_dim,
            int(fusion_config.head_hidden),
            self.output_dim,
            norm=fusion_config.head_norm,
            activation=fusion_config.head_activation,
            dropout=float(fusion_config.head_dropout),
            norm_eps=float(fusion_config.head_norm_eps),
            batch_momentum=float(fusion_config.head_bn_momentum),
        )
        self.weight_init = _validate_choice(
            "fusion.weight_init", fusion_config.weight_init, {"pytorch", "saluki"}
        )
        if self.weight_init == "saluki":
            self._reset_fusion_parameters()

    @staticmethod
    def _validate_configs(
        saluki: _SalukiBranchConfig,
        cnn: _CNNBranchConfig,
        fusion: _FusionConfig,
    ) -> None:
        positive = {
            "saluki.seq_depth": saluki.seq_depth,
            "saluki.filters": saluki.filters,
            "saluki.kernel_size": saluki.kernel_size,
            "saluki.num_layers": saluki.num_layers,
            "cnn.in_ch": cnn.in_ch,
            "cnn.n_filters": cnn.n_filters,
            "cnn.kernel_size": cnn.kernel_size,
            "cnn.n_layers": cnn.n_layers,
            "fusion.saluki_projection_dim": fusion.saluki_projection_dim,
            "fusion.cnn_projection_dim": fusion.cnn_projection_dim,
            "fusion.head_hidden": fusion.head_hidden,
            "fusion.output_dim": fusion.output_dim,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(saluki.seq_depth) != int(cnn.in_ch):
            raise ValueError(
                "saluki.seq_depth and cnn.in_ch must match because both branches receive "
                "the same input"
            )
        if int(saluki.augment_shift) < 0:
            raise ValueError("saluki.augment_shift must be non-negative")
        if float(saluki.ln_epsilon) <= 0 or float(fusion.head_norm_eps) <= 0:
            raise ValueError("normalization epsilons must be positive")
        _validate_probability("saluki.dropout", saluki.dropout)
        _validate_probability("cnn.dropout", cnn.dropout)
        _validate_probability("fusion.projection_dropout", fusion.projection_dropout)
        _validate_probability("fusion.head_dropout", fusion.head_dropout)
        _validate_choice("saluki.pooling", saluki.pooling, {"max", "average"})
        _validate_choice("fusion.mode", fusion.mode, {"concat", "project_concat"})
        _validate_choice(
            "fusion.projection_norm", fusion.projection_norm, {"none", "batch", "layer"}
        )
        _validate_choice(
            "fusion.projection_activation",
            fusion.projection_activation,
            {"none", "relu", "gelu", "silu"},
        )
        _validate_choice("fusion.head_norm", fusion.head_norm, {"none", "batch", "layer"})
        _validate_choice(
            "fusion.head_activation", fusion.head_activation, {"relu", "gelu", "silu"}
        )
        _validate_choice("fusion.weight_init", fusion.weight_init, {"pytorch", "saluki"})
        if not 0.0 < float(fusion.head_bn_momentum) <= 1.0:
            raise ValueError("fusion.head_bn_momentum must be in (0, 1]")

    def _reset_fusion_parameters(self) -> None:
        for projection in (self.saluki_projection, self.cnn_projection):
            if isinstance(projection, _FeatureProjection):
                keras_he_normal_trunc_(projection.linear.weight)
                nn.init.zeros_(projection.linear.bias)
        for linear in (self.head.linear1, self.head.linear2):
            keras_he_normal_trunc_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected a three-dimensional input, got shape {tuple(x.shape)}")
        if x.shape[1] == self.in_ch:
            return x.float()
        if x.shape[-1] == self.in_ch:
            return x.transpose(1, 2).float()
        raise ValueError(f"Expected channel depth {self.in_ch}, got shape {tuple(x.shape)}")

    def forward_branch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Saluki and SmallCNN embeddings for the same shifted input."""

        shared = self.shift(self._to_bcl(x))
        return self.saluki_encoder(shared), self.cnn_encoder(shared)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the concatenated embedding consumed by the final head."""

        saluki_features, cnn_features = self.forward_branch_features(x)
        return torch.cat(
            [self.saluki_projection(saluki_features), self.cnn_projection(cnn_features)],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return hybrid predictions for channel-first or channel-last inputs."""

        prediction = self.head(self.forward_features(x))
        return prediction[:, 0] if self.output_dim == 1 else prediction
