from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from transcriptml.models.common import ChannelLayerNorm, dropout_or_identity
from transcriptml.models.reproduce import StochasticShift, keras_he_normal_trunc_


IntPerLayer = int | list[int] | tuple[int, ...]
StrPerLayer = str | list[str] | tuple[str, ...]


@dataclass
class ConvolutionUniverseConfig:
    """Flat, JSON-serializable configuration for :class:`ConvolutionUniverse`."""

    in_ch: int = 6
    output_dim: int = 1
    augment_shift: int = 0
    weight_init: str = "pytorch"

    filters: IntPerLayer = 64
    num_layers: int = 6
    stem_kernel_size: int = 11
    stem_padding: str = "same"
    stem_bias: bool = True

    block_type: str = "conv"
    kernel_size: IntPerLayer = 7
    dilations: IntPerLayer = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    padding: str = "same"
    conv_bias: bool = True
    norm: str = "layer"
    norm_position: str = "post"
    norm_eps: float = 1e-5
    activation: str = "gelu"
    dropout: float = 0.1
    use_residual: bool = True
    pooling: StrPerLayer = "none"
    pool_size: IntPerLayer = 2

    mlp_ratio: int = 4
    layer_scale_init: float | None = 1e-6
    drop_path_rate: float = 0.0

    pre_rnn_norm: str = "none"
    pre_rnn_activation: str = "none"
    rnn_type: str = "none"
    rnn_hidden: int = 128
    rnn_layers: int = 1
    rnn_bidirectional: bool = False
    rnn_reverse: bool = False
    rnn_summary: str = "last"
    rnn_dropout: float = 0.0

    global_pool: str = "avgmax"
    global_pool_size: int = 1

    head_type: str = "mlp"
    head_hidden: int | list[int] | tuple[int, ...] | None = 64
    head_norm: str = "layer"
    head_norm_position: str = "post"
    head_activation: str = "gelu"
    head_dropout: float = 0.1
    head_norm_eps: float = 1e-5
    head_bn_momentum: float = 0.1

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``ConvolutionUniverse``."""

        return asdict(self)

    @classmethod
    def saluki(cls) -> "ConvolutionUniverseConfig":
        """Return the canonical configuration equivalent to ``SalukiExact``."""

        return cls(
            in_ch=6,
            output_dim=1,
            augment_shift=3,
            weight_init="saluki",
            filters=64,
            num_layers=6,
            stem_kernel_size=5,
            stem_padding="valid",
            stem_bias=False,
            block_type="conv",
            kernel_size=5,
            dilations=1,
            padding="valid",
            conv_bias=True,
            norm="layer",
            norm_position="pre",
            norm_eps=0.007,
            activation="relu",
            dropout=0.3,
            use_residual=False,
            pooling="max",
            pool_size=2,
            mlp_ratio=4,
            layer_scale_init=1e-6,
            drop_path_rate=0.0,
            pre_rnn_norm="layer",
            pre_rnn_activation="relu",
            rnn_type="gru",
            rnn_hidden=64,
            rnn_layers=1,
            rnn_bidirectional=False,
            rnn_reverse=True,
            rnn_summary="last",
            rnn_dropout=0.0,
            global_pool="avgmax",
            global_pool_size=1,
            head_type="mlp",
            head_hidden=64,
            head_norm="batch",
            head_norm_position="pre",
            head_activation="relu",
            head_dropout=0.3,
            head_norm_eps=0.001,
            head_bn_momentum=0.1,
        )


def _validate_choice(name: str, value: str, choices: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return normalized


def _validate_probability(name: str, value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")
    return probability


def _expand_int(value: IntPerLayer, depth: int, name: str) -> list[int]:
    if isinstance(value, int):
        values = [int(value)] * depth
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
        if len(values) != depth:
            raise ValueError(f"{name} must have exactly num_layers={depth} entries, got {len(values)}")
    else:
        raise TypeError(f"{name} must be an integer or a sequence of integers")
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} values must all be positive, got {values}")
    return values


def _canonical_pooling(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "average":
        normalized = "avg"
    return _validate_choice("pooling", normalized, {"none", "max", "avg"})


def _expand_pooling(value: StrPerLayer, depth: int) -> list[str]:
    if isinstance(value, str):
        return [_canonical_pooling(value)] * depth
    if isinstance(value, Sequence):
        values = [_canonical_pooling(item) for item in value]
        if len(values) != depth:
            raise ValueError(
                f"pooling must have exactly num_layers={depth} entries, got {len(values)}"
            )
        return values
    raise TypeError("pooling must be a string or a sequence of strings")


def _padding_for(kernel_size: int, dilation: int, padding: str, name: str) -> int:
    if padding == "valid":
        return 0
    if kernel_size % 2 == 0:
        raise ValueError(f"{name} must be odd when padding='same', got {kernel_size}")
    return dilation * (kernel_size - 1) // 2


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


def _channel_norm(kind: str, channels: int, eps: float) -> nn.Module:
    normalized = _validate_choice("norm", kind, {"none", "batch", "layer"})
    if normalized == "batch":
        return nn.BatchNorm1d(channels, eps=float(eps))
    if normalized == "layer":
        return ChannelLayerNorm(channels, eps=float(eps))
    return nn.Identity()


def _feature_norm(kind: str, features: int, eps: float, momentum: float) -> nn.Module:
    normalized = _validate_choice("head_norm", kind, {"none", "batch", "layer"})
    if normalized == "batch":
        return nn.BatchNorm1d(features, eps=float(eps), momentum=float(momentum))
    if normalized == "layer":
        return nn.LayerNorm(features, eps=float(eps))
    return nn.Identity()


class DropPath(nn.Module):
    """Per-example stochastic depth for residual branches."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = _validate_probability("drop_path probability", probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class StandardConvBlock1D(nn.Module):
    """Configurable pre- or post-normalized Conv1d block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        padding: str,
        bias: bool,
        norm: str,
        norm_position: str,
        norm_eps: float,
        activation: str,
        dropout: float,
        use_residual: bool,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_mode = padding
        self.pad = _padding_for(self.kernel_size, self.dilation, padding, "kernel_size")
        self.norm_position = _validate_choice("norm_position", norm_position, {"pre", "post"})
        self.use_residual = bool(use_residual)
        if self.use_residual and padding != "same":
            raise ValueError("use_residual=True requires padding='same'")

        self.norm = _channel_norm(
            norm,
            in_channels if self.norm_position == "pre" else out_channels,
            norm_eps,
        )
        self.activation = _activation(activation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=self.pad,
            dilation=self.dilation,
            bias=bool(bias),
        )
        self.dropout = dropout_or_identity(dropout)
        if not self.use_residual:
            self.shortcut = None
        elif in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.norm_position == "pre":
            z = self.conv(self.activation(self.norm(x)))
            z = self.dropout(z)
        else:
            z = self.dropout(self.activation(self.norm(self.conv(x))))
        if self.shortcut is not None:
            z = z + self.shortcut(residual)
        return z


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt-v1-style depthwise residual block for one-dimensional inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        padding: str,
        bias: bool,
        mlp_ratio: int,
        layer_scale_init: float | None,
        drop_path: float,
        use_residual: bool,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_mode = padding
        self.pad = _padding_for(self.kernel_size, self.dilation, padding, "kernel_size")
        self.use_residual = bool(use_residual)
        if self.use_residual and padding != "same":
            raise ValueError("use_residual=True requires padding='same'")
        if int(mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be at least 1")

        self.transition = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.depthwise = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=self.pad,
            dilation=self.dilation,
            groups=out_channels,
            bias=bool(bias),
        )
        hidden = out_channels * int(mlp_ratio)
        self.norm = nn.LayerNorm(out_channels)
        self.pointwise1 = nn.Linear(out_channels, hidden, bias=bool(bias))
        self.activation = nn.GELU()
        self.pointwise2 = nn.Linear(hidden, out_channels, bias=bool(bias))
        self.gamma = (
            None
            if layer_scale_init is None
            else nn.Parameter(float(layer_scale_init) * torch.ones(out_channels))
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.transition(x)
        z = self.depthwise(base).transpose(1, 2)
        z = self.pointwise2(self.activation(self.pointwise1(self.norm(z))))
        if self.gamma is not None:
            z = z * self.gamma.view(1, 1, -1)
        z = self.drop_path(z.transpose(1, 2))
        return base + z if self.use_residual else z


class ConvTower1D(nn.Module):
    """Stem plus repeated convolutional blocks and optional per-layer pooling."""

    def __init__(
        self,
        *,
        in_ch: int,
        filters: list[int],
        stem_kernel_size: int,
        stem_padding: str,
        stem_bias: bool,
        block_type: str,
        kernel_sizes: list[int],
        dilations: list[int],
        padding: str,
        conv_bias: bool,
        norm: str,
        norm_position: str,
        norm_eps: float,
        activation: str,
        dropout: float,
        use_residual: bool,
        pooling: list[str],
        pool_sizes: list[int],
        mlp_ratio: int,
        layer_scale_init: float | None,
        drop_path_rate: float,
    ):
        super().__init__()
        self.stem_kernel_size = int(stem_kernel_size)
        self.stem_padding_mode = stem_padding
        self.stem_pad = _padding_for(self.stem_kernel_size, 1, stem_padding, "stem_kernel_size")
        self.kernel_sizes = list(kernel_sizes)
        self.dilations = list(dilations)
        self.pooling = list(pooling)
        self.pool_sizes = list(pool_sizes)
        self.padding_mode = padding

        self.stem = nn.Conv1d(
            int(in_ch),
            filters[0],
            kernel_size=self.stem_kernel_size,
            padding=self.stem_pad,
            bias=bool(stem_bias),
        )
        self.stages = nn.ModuleList()
        depth = len(filters)
        in_channels = filters[0]
        for index, out_channels in enumerate(filters):
            drop_path = float(drop_path_rate) if depth == 1 else float(drop_path_rate) * index / (depth - 1)
            if block_type == "conv":
                block: nn.Module = StandardConvBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[index],
                    dilation=dilations[index],
                    padding=padding,
                    bias=conv_bias,
                    norm=norm,
                    norm_position=norm_position,
                    norm_eps=norm_eps,
                    activation=activation,
                    dropout=dropout,
                    use_residual=use_residual,
                )
            else:
                block = ConvNeXtBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[index],
                    dilation=dilations[index],
                    padding=padding,
                    bias=conv_bias,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init=layer_scale_init,
                    drop_path=drop_path,
                    use_residual=use_residual,
                )
            pool_kind = pooling[index]
            if pool_kind == "max":
                pool: nn.Module = nn.MaxPool1d(pool_sizes[index])
            elif pool_kind == "avg":
                pool = nn.AvgPool1d(pool_sizes[index])
            else:
                pool = nn.Identity()
            self.stages.append(nn.ModuleDict({"block": block, "pool": pool}))
            in_channels = out_channels

        self.out_channels = filters[-1]

    def output_length(self, input_length: int) -> int:
        length = int(input_length)
        if length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if self.stem_padding_mode == "valid":
            length -= self.stem_kernel_size - 1
        if length <= 0:
            raise ValueError("sequence length collapses in the convolutional stem")
        for index, (kernel, dilation, pool, pool_size) in enumerate(
            zip(self.kernel_sizes, self.dilations, self.pooling, self.pool_sizes)
        ):
            if self.padding_mode == "valid":
                length -= dilation * (kernel - 1)
            if length <= 0:
                raise ValueError(f"sequence length collapses in convolutional block {index}")
            if pool != "none":
                length = (length - pool_size) // pool_size + 1
                if length <= 0:
                    raise ValueError(f"sequence length collapses in pooling layer {index}")
        return length

    @property
    def receptive_field(self) -> int:
        receptive_field = 1 + (self.stem_kernel_size - 1)
        jump = 1
        for kernel, dilation, pool, pool_size in zip(
            self.kernel_sizes, self.dilations, self.pooling, self.pool_sizes
        ):
            receptive_field += (kernel - 1) * dilation * jump
            if pool != "none":
                receptive_field += (pool_size - 1) * jump
                jump *= pool_size
        return receptive_field

    @property
    def downsample_factor(self) -> int:
        factor = 1
        for pool, pool_size in zip(self.pooling, self.pool_sizes):
            if pool != "none":
                factor *= pool_size
        return factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        for stage in self.stages:
            z = stage["pool"](stage["block"](z))
        return z


class DenseHead(nn.Module):
    """Linear or normalized MLP prediction head."""

    def __init__(
        self,
        in_features: int,
        output_dim: int,
        *,
        head_type: str,
        hidden: list[int],
        norm: str,
        norm_position: str,
        activation: str,
        dropout: float,
        norm_eps: float,
        batch_momentum: float,
    ):
        super().__init__()
        self.head_type = head_type
        self.norm_position = norm_position
        if head_type == "linear":
            self.linears = nn.ModuleList([nn.Linear(in_features, output_dim)])
            self.norms = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            return

        dimensions = [in_features, *hidden, output_dim]
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        if norm_position == "pre":
            for index, (current, following) in enumerate(zip(dimensions[:-1], dimensions[1:])):
                self.norms.append(_feature_norm(norm, current, norm_eps, batch_momentum))
                self.activations.append(_activation(activation))
                self.linears.append(nn.Linear(current, following))
                self.dropouts.append(
                    dropout_or_identity(dropout) if index < len(dimensions) - 2 else nn.Identity()
                )
        else:
            for index, (current, following) in enumerate(zip(dimensions[:-1], dimensions[1:])):
                self.linears.append(nn.Linear(current, following))
                if index < len(dimensions) - 2:
                    self.norms.append(_feature_norm(norm, following, norm_eps, batch_momentum))
                    self.activations.append(_activation(activation))
                    self.dropouts.append(dropout_or_identity(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "linear":
            return self.linears[0](x)
        z = x
        if self.norm_position == "pre":
            for norm, activation, linear, dropout in zip(
                self.norms, self.activations, self.linears, self.dropouts
            ):
                z = dropout(linear(activation(norm(z))))
            return z
        for index, linear in enumerate(self.linears):
            z = linear(z)
            if index < len(self.linears) - 1:
                z = self.dropouts[index](self.activations[index](self.norms[index](z)))
        return z


class ConvolutionUniverse(nn.Module):
    """Flexible CNN or CNN-RNN model for fixed-width RNA sequence tensors."""

    def __init__(
        self,
        in_ch: int = 6,
        output_dim: int = 1,
        augment_shift: int = 0,
        weight_init: str = "pytorch",
        filters: IntPerLayer = 64,
        num_layers: int = 6,
        stem_kernel_size: int = 11,
        stem_padding: str = "same",
        stem_bias: bool = True,
        block_type: str = "conv",
        kernel_size: IntPerLayer = 7,
        dilations: IntPerLayer = (1, 2, 4, 8, 16, 32),
        padding: str = "same",
        conv_bias: bool = True,
        norm: str = "layer",
        norm_position: str = "post",
        norm_eps: float = 1e-5,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_residual: bool = True,
        pooling: StrPerLayer = "none",
        pool_size: IntPerLayer = 2,
        mlp_ratio: int = 4,
        layer_scale_init: float | None = 1e-6,
        drop_path_rate: float = 0.0,
        pre_rnn_norm: str = "none",
        pre_rnn_activation: str = "none",
        rnn_type: str = "none",
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = False,
        rnn_reverse: bool = False,
        rnn_summary: str = "last",
        rnn_dropout: float = 0.0,
        global_pool: str = "avgmax",
        global_pool_size: int = 1,
        head_type: str = "mlp",
        head_hidden: int | list[int] | tuple[int, ...] | None = 64,
        head_norm: str = "layer",
        head_norm_position: str = "post",
        head_activation: str = "gelu",
        head_dropout: float = 0.1,
        head_norm_eps: float = 1e-5,
        head_bn_momentum: float = 0.1,
    ):
        super().__init__()
        if int(in_ch) <= 0:
            raise ValueError("in_ch must be positive")
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")
        if int(augment_shift) < 0:
            raise ValueError("augment_shift must be non-negative")
        if float(norm_eps) <= 0 or float(head_norm_eps) <= 0:
            raise ValueError("normalization epsilons must be positive")

        self.in_ch = int(in_ch)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.weight_init = _validate_choice("weight_init", weight_init, {"pytorch", "saluki"})
        block_type = _validate_choice("block_type", block_type, {"conv", "convnext"})
        stem_padding = _validate_choice("stem_padding", stem_padding, {"same", "valid"})
        padding = _validate_choice("padding", padding, {"same", "valid"})
        norm = _validate_choice("norm", norm, {"none", "batch", "layer"})
        norm_position = _validate_choice("norm_position", norm_position, {"pre", "post"})
        activation = _validate_choice("activation", activation, {"relu", "gelu", "silu"})
        self.rnn_type = _validate_choice("rnn_type", rnn_type, {"none", "gru", "lstm"})
        self.rnn_summary = _validate_choice("rnn_summary", rnn_summary, {"last", "mean", "max"})
        self.global_pool = _validate_choice("global_pool", global_pool, {"avg", "max", "avgmax"})
        head_type = _validate_choice("head_type", head_type, {"linear", "mlp"})
        head_norm = _validate_choice("head_norm", head_norm, {"none", "batch", "layer"})
        head_norm_position = _validate_choice(
            "head_norm_position", head_norm_position, {"pre", "post"}
        )
        head_activation = _validate_choice(
            "head_activation", head_activation, {"relu", "gelu", "silu"}
        )
        if not 0.0 < float(head_bn_momentum) <= 1.0:
            raise ValueError("head_bn_momentum must be in (0, 1]")
        dropout = _validate_probability("dropout", dropout)
        drop_path_rate = _validate_probability("drop_path_rate", drop_path_rate)
        rnn_dropout = _validate_probability("rnn_dropout", rnn_dropout)
        head_dropout = _validate_probability("head_dropout", head_dropout)
        if int(global_pool_size) < 1:
            raise ValueError("global_pool_size must be at least 1")
        if int(rnn_hidden) < 1 or int(rnn_layers) < 1:
            raise ValueError("rnn_hidden and rnn_layers must be positive")
        if int(mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be at least 1")

        expanded_filters = _expand_int(filters, self.num_layers, "filters")
        expanded_kernels = _expand_int(kernel_size, self.num_layers, "kernel_size")
        expanded_dilations = _expand_int(dilations, self.num_layers, "dilations")
        expanded_pooling = _expand_pooling(pooling, self.num_layers)
        expanded_pool_sizes = _expand_int(pool_size, self.num_layers, "pool_size")
        if int(stem_kernel_size) <= 0:
            raise ValueError("stem_kernel_size must be positive")
        _padding_for(int(stem_kernel_size), 1, stem_padding, "stem_kernel_size")
        for kernel, dilation in zip(expanded_kernels, expanded_dilations):
            _padding_for(kernel, dilation, padding, "kernel_size")
        if bool(use_residual) and padding != "same":
            raise ValueError("use_residual=True requires padding='same'")

        if head_hidden is None:
            hidden: list[int] = []
        elif isinstance(head_hidden, int):
            hidden = [int(head_hidden)]
        elif isinstance(head_hidden, Sequence) and not isinstance(head_hidden, (str, bytes)):
            hidden = [int(item) for item in head_hidden]
        else:
            raise TypeError("head_hidden must be an integer, a sequence of integers, or None")
        if any(item <= 0 for item in hidden):
            raise ValueError("head_hidden values must be positive")

        self.shift = StochasticShift(int(augment_shift))
        self.tower = ConvTower1D(
            in_ch=self.in_ch,
            filters=expanded_filters,
            stem_kernel_size=int(stem_kernel_size),
            stem_padding=stem_padding,
            stem_bias=bool(stem_bias),
            block_type=block_type,
            kernel_sizes=expanded_kernels,
            dilations=expanded_dilations,
            padding=padding,
            conv_bias=bool(conv_bias),
            norm=norm,
            norm_position=norm_position,
            norm_eps=float(norm_eps),
            activation=activation,
            dropout=dropout,
            use_residual=bool(use_residual),
            pooling=expanded_pooling,
            pool_sizes=expanded_pool_sizes,
            mlp_ratio=int(mlp_ratio),
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
        )

        pre_rnn_norm = _validate_choice(
            "pre_rnn_norm", pre_rnn_norm, {"none", "batch", "layer"}
        )
        pre_rnn_activation = _validate_choice(
            "pre_rnn_activation", pre_rnn_activation, {"none", "relu", "gelu", "silu"}
        )
        self.pre_rnn_norm = _channel_norm(pre_rnn_norm, self.tower.out_channels, norm_eps)
        self.pre_rnn_activation = _activation(pre_rnn_activation, allow_none=True)
        self.rnn_reverse = bool(rnn_reverse)
        self.rnn_layers = int(rnn_layers)
        self.rnn_bidirectional = bool(rnn_bidirectional)
        if self.rnn_type == "none":
            self.rnn: nn.GRU | nn.LSTM | None = None
            pool_branches = 2 if self.global_pool == "avgmax" else 1
            feature_dim = self.tower.out_channels * int(global_pool_size) * pool_branches
        else:
            rnn_class = nn.GRU if self.rnn_type == "gru" else nn.LSTM
            self.rnn = rnn_class(
                input_size=self.tower.out_channels,
                hidden_size=int(rnn_hidden),
                num_layers=self.rnn_layers,
                batch_first=True,
                bidirectional=self.rnn_bidirectional,
                dropout=rnn_dropout if self.rnn_layers > 1 else 0.0,
            )
            feature_dim = int(rnn_hidden) * (2 if self.rnn_bidirectional else 1)

        self.global_pool_size = int(global_pool_size)
        self.head = DenseHead(
            feature_dim,
            self.output_dim,
            head_type=head_type,
            hidden=hidden,
            norm=head_norm,
            norm_position=head_norm_position,
            activation=head_activation,
            dropout=head_dropout,
            norm_eps=float(head_norm_eps),
            batch_momentum=float(head_bn_momentum),
        )
        if self.weight_init == "saluki":
            self.reset_saluki_parameters()

    def reset_saluki_parameters(self) -> None:
        """Initialize compatible layers using the legacy Saluki conventions."""

        keras_he_normal_trunc_(self.tower.stem.weight)
        if self.tower.stem.bias is not None:
            nn.init.zeros_(self.tower.stem.bias)
        for stage in self.tower.stages:
            block = stage["block"]
            if isinstance(block, StandardConvBlock1D):
                keras_he_normal_trunc_(block.conv.weight)
                if block.conv.bias is not None:
                    nn.init.zeros_(block.conv.bias)
                if isinstance(block.shortcut, nn.Conv1d):
                    keras_he_normal_trunc_(block.shortcut.weight)
            elif isinstance(block, ConvNeXtBlock1D):
                if isinstance(block.transition, nn.Conv1d):
                    keras_he_normal_trunc_(block.transition.weight)
                keras_he_normal_trunc_(block.depthwise.weight)
                keras_he_normal_trunc_(block.pointwise1.weight)
                keras_he_normal_trunc_(block.pointwise2.weight)
                for layer in (block.depthwise, block.pointwise1, block.pointwise2):
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if self.rnn is not None:
            for name, parameter in self.rnn.named_parameters():
                if "weight_ih" in name:
                    keras_he_normal_trunc_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        for linear in self.head.linears:
            keras_he_normal_trunc_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected a three-dimensional input, got shape {tuple(x.shape)}")
        if x.shape[1] == self.in_ch:
            return x.float()
        if x.shape[-1] == self.in_ch:
            return x.transpose(1, 2).float()
        raise ValueError(f"Expected channel depth {self.in_ch}, got shape {tuple(x.shape)}")

    def output_length(self, input_length: int) -> int:
        """Return the convolutional tower length for an input sequence length."""

        return self.tower.output_length(input_length)

    @property
    def receptive_field(self) -> int:
        """Receptive-field width of one position at the tower output."""

        return self.tower.receptive_field

    @property
    def downsample_factor(self) -> int:
        """Product of all configured internal pooling strides."""

        return self.tower.downsample_factor

    def _pool_convolutional_features(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.global_pool in {"avg", "avgmax"}:
            features.append(F.adaptive_avg_pool1d(x, self.global_pool_size).flatten(1))
        if self.global_pool in {"max", "avgmax"}:
            features.append(F.adaptive_max_pool1d(x, self.global_pool_size).flatten(1))
        return torch.cat(features, dim=1)

    def _summarize_rnn(
        self,
        output: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.rnn_summary == "mean":
            return output.mean(dim=1)
        if self.rnn_summary == "max":
            return output.amax(dim=1)
        hidden = state[0] if self.rnn_type == "lstm" else state
        directions = 2 if self.rnn_bidirectional else 1
        hidden = hidden.view(self.rnn_layers, directions, hidden.shape[1], hidden.shape[2])
        final = hidden[-1]
        return final[0] if directions == 1 else torch.cat([final[0], final[1]], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input into the feature vector consumed by the dense head."""

        z = self.shift(self._to_bcl(x))
        self.output_length(z.shape[-1])
        z = self.tower(z)
        if self.rnn is None:
            return self._pool_convolutional_features(z)
        z = self.pre_rnn_activation(self.pre_rnn_norm(z)).transpose(1, 2)
        if self.rnn_reverse:
            z = torch.flip(z, dims=[1])
        output, state = self.rnn(z)
        return self._summarize_rnn(output, state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for a batch of channel-first or channel-last inputs."""

        prediction = self.head(self.forward_features(x))
        return prediction[:, 0] if self.output_dim == 1 else prediction
