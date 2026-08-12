"""Configurable sequence-only RBPNet profile and enrichment model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transcriptml.models.common import ChannelLayerNorm, dropout_or_identity


@dataclass
class RBPNetConfig:
    """Configuration for the first TranscriptML RBPNet model family."""

    in_ch: int = 4
    n_filters: int = 128
    initial_kernel_size: int = 12
    n_residual_blocks: int = 5
    residual_kernel_size: int = 6
    dilations: list[int] | None = None
    normalization: str = "batch"
    dropout: float = 0.25
    initial_bias: bool = False
    residual_bias: bool = True
    profile_head_type: str = "transpose_conv"
    profile_head_kernel_size: int = 25
    profile_head_bias: bool = True
    enrichment_head_type: str = "none"
    enrichment_hidden: int = 64
    enrichment_dropout: float = 0.0
    profile_length: int | None = 300
    batch_norm_eps: float = 1e-5
    batch_norm_momentum: float = 0.1

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor arguments accepted by :class:`RBPNet`."""

        return asdict(self)


@dataclass(frozen=True)
class RBPNetOutput:
    """Structured differentiable outputs from :class:`RBPNet`."""

    target_logits: torch.Tensor
    control_logits: torch.Tensor
    target_log_probs: torch.Tensor
    control_log_probs: torch.Tensor
    target_probs: torch.Tensor
    control_probs: torch.Tensor
    mixing_logit: torch.Tensor
    pi: torch.Tensor
    ip_log_probs: torch.Tensor
    ip_probs: torch.Tensor
    enrichment_logit: torch.Tensor | None = None


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def resolve_dilations(n_residual_blocks: int, dilations: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    """Resolve an explicit schedule or powers-of-two defaults starting at two."""

    n_blocks = int(n_residual_blocks)
    if n_blocks < 0:
        raise ValueError("n_residual_blocks must be non-negative")
    values = tuple(2 ** (index + 1) for index in range(n_blocks)) if dilations is None else tuple(
        int(value) for value in dilations
    )
    if len(values) != n_blocks:
        raise ValueError("dilations length must equal n_residual_blocks")
    if any(value <= 0 for value in values):
        raise ValueError("all dilations must be positive")
    return values


def theoretical_receptive_field(
    initial_kernel_size: int,
    residual_kernel_size: int,
    dilations: list[int] | tuple[int, ...],
) -> int:
    """Return the position-preserving trunk's theoretical receptive-field width."""

    initial = _validate_positive_int("initial_kernel_size", initial_kernel_size)
    residual = _validate_positive_int("residual_kernel_size", residual_kernel_size)
    return 1 + (initial - 1) + sum((residual - 1) * int(dilation) for dilation in dilations)


def same_padding(kernel_size: int, dilation: int = 1) -> tuple[int, int]:
    """Return explicit left/right padding used to preserve indexed length.

    Even effective kernels necessarily have a half-base geometric center. The
    extra base is placed on the right, matching the usual ``same`` convention;
    output index ``i`` nevertheless remains output index ``i`` at every layer.
    """

    kernel = _validate_positive_int("kernel_size", kernel_size)
    dilation = _validate_positive_int("dilation", dilation)
    total = dilation * (kernel - 1)
    left = total // 2
    return left, total - left


class SamePadConv1d(nn.Module):
    """Conv1d with explicit, version-stable asymmetric same padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = same_padding(kernel_size, dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.padding
        return self.conv(F.pad(x, (left, right)))


class SameLengthConvTranspose1d(nn.Module):
    """Stride-one transposed convolution cropped to the indexed input length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.crop = same_padding(kernel_size)
        self.conv = nn.ConvTranspose1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        left, _ = self.crop
        return output[..., left : left + x.shape[-1]]


def _normalization(
    kind: str,
    channels: int,
    *,
    eps: float,
    momentum: float,
) -> nn.Module:
    kind = str(kind).strip().lower()
    if kind in {"batch", "batchnorm", "batch_norm"}:
        return nn.BatchNorm1d(int(channels), eps=float(eps), momentum=float(momentum))
    if kind in {"layer", "layernorm", "layer_norm"}:
        return ChannelLayerNorm(int(channels), eps=float(eps))
    if kind in {"none", "identity", "off"}:
        return nn.Identity()
    raise ValueError("normalization must be one of: batch, layer, none")


class DilatedResidualBlock(nn.Module):
    """RBPNet-style dilated convolution, normalization, ReLU, dropout, add."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        *,
        normalization: str,
        dropout: float,
        bias: bool,
        batch_norm_eps: float,
        batch_norm_momentum: float,
    ) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv = SamePadConv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.norm = _normalization(
            normalization,
            channels,
            eps=batch_norm_eps,
            momentum=batch_norm_momentum,
        )
        self.activation = nn.ReLU()
        self.dropout = dropout_or_identity(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(self.activation(self.norm(self.conv(x))))
        if residual.shape != x.shape:
            raise RuntimeError("residual convolution changed positional shape")
        return x + residual


class PositionalProfileHead(nn.Module):
    """Modular one-channel positional-logit head."""

    def __init__(
        self,
        channels: int,
        *,
        head_type: str,
        kernel_size: int,
        bias: bool,
    ) -> None:
        super().__init__()
        head_type = str(head_type).strip().lower()
        if head_type in {"transpose_conv", "transposed_conv", "conv_transpose"}:
            self.layer = SameLengthConvTranspose1d(
                channels, 1, kernel_size, bias=bias
            )
            self.head_type = "transpose_conv"
        elif head_type in {"conv", "convolution"}:
            self.layer = SamePadConv1d(channels, 1, kernel_size, bias=bias)
            self.head_type = "conv"
        else:
            raise ValueError("profile_head_type must be transpose_conv or conv")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden).squeeze(1)


def _masked_log_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, log_probs.exp()
    mask = torch.as_tensor(mask, device=logits.device).bool()
    if mask.shape != logits.shape:
        raise ValueError(
            f"profile_mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}"
        )
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every profile_mask row must contain at least one valid position")
    normalized = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    log_probs = torch.where(mask, normalized, torch.zeros_like(normalized))
    probs = torch.where(mask, normalized.exp(), torch.zeros_like(normalized))
    return log_probs, probs


class RBPNet(nn.Module):
    """Sequence-only RBPNet with latent target/control mixture and optional eta."""

    def __init__(
        self,
        in_ch: int = 4,
        n_filters: int = 128,
        initial_kernel_size: int = 12,
        n_residual_blocks: int = 5,
        residual_kernel_size: int = 6,
        dilations: list[int] | tuple[int, ...] | None = None,
        normalization: str = "batch",
        dropout: float = 0.25,
        initial_bias: bool = False,
        residual_bias: bool = True,
        profile_head_type: str = "transpose_conv",
        profile_head_kernel_size: int = 25,
        profile_head_bias: bool = True,
        enrichment_head_type: str = "none",
        enrichment_hidden: int = 64,
        enrichment_dropout: float = 0.0,
        profile_length: int | None = 300,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_ch = _validate_positive_int("in_ch", in_ch)
        self.n_filters = _validate_positive_int("n_filters", n_filters)
        self.initial_kernel_size = _validate_positive_int(
            "initial_kernel_size", initial_kernel_size
        )
        self.residual_kernel_size = _validate_positive_int(
            "residual_kernel_size", residual_kernel_size
        )
        self.dilations = resolve_dilations(n_residual_blocks, dilations)
        self.profile_length = None if profile_length is None else _validate_positive_int(
            "profile_length", profile_length
        )
        if not 0 <= float(dropout) < 1 or not 0 <= float(enrichment_dropout) < 1:
            raise ValueError("dropout probabilities must be in [0, 1)")

        self.initial_conv = SamePadConv1d(
            self.in_ch,
            self.n_filters,
            self.initial_kernel_size,
            bias=initial_bias,
        )
        self.initial_activation = nn.ReLU()
        self.residual_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    self.n_filters,
                    self.residual_kernel_size,
                    dilation,
                    normalization=normalization,
                    dropout=dropout,
                    bias=residual_bias,
                    batch_norm_eps=batch_norm_eps,
                    batch_norm_momentum=batch_norm_momentum,
                )
                for dilation in self.dilations
            ]
        )
        profile_kwargs = {
            "head_type": profile_head_type,
            "kernel_size": profile_head_kernel_size,
            "bias": profile_head_bias,
        }
        self.target_profile_head = PositionalProfileHead(self.n_filters, **profile_kwargs)
        self.control_profile_head = PositionalProfileHead(self.n_filters, **profile_kwargs)
        self.mixing_head = nn.Linear(self.n_filters, 1)

        enrichment_kind = str(enrichment_head_type).strip().lower()
        if enrichment_kind in {"none", "off", "disabled"}:
            self.enrichment_head: nn.Module | None = None
            self.enrichment_head_type = "none"
        elif enrichment_kind == "linear":
            self.enrichment_head = nn.Linear(self.n_filters, 1)
            self.enrichment_head_type = "linear"
        elif enrichment_kind == "mlp":
            hidden = _validate_positive_int("enrichment_hidden", enrichment_hidden)
            self.enrichment_head = nn.Sequential(
                nn.Linear(self.n_filters, hidden),
                nn.ReLU(),
                dropout_or_identity(float(enrichment_dropout)),
                nn.Linear(hidden, 1),
            )
            self.enrichment_head_type = "mlp"
        else:
            raise ValueError("enrichment_head_type must be one of: none, linear, mlp")

    @property
    def receptive_field(self) -> int:
        """Theoretical receptive-field width of the shared trunk."""

        return theoretical_receptive_field(
            self.initial_kernel_size,
            self.residual_kernel_size,
            self.dilations,
        )

    @property
    def receptive_field_extents(self) -> tuple[int, int]:
        """Return left/right trunk context extents under explicit same padding."""

        left, right = same_padding(self.initial_kernel_size)
        for dilation in self.dilations:
            block_left, block_right = same_padding(
                self.residual_kernel_size, dilation
            )
            left += block_left
            right += block_right
        return left, right

    @property
    def enrichment_enabled(self) -> bool:
        return self.enrichment_head is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared position-preserving hidden representation."""

        if x.ndim != 3 or x.shape[1] != self.in_ch:
            raise ValueError(
                f"RBPNet expects (B,{self.in_ch},L), received {tuple(x.shape)}"
            )
        if self.profile_length is not None and x.shape[-1] != self.profile_length:
            raise ValueError(
                f"configured profile_length is {self.profile_length}, received {x.shape[-1]}"
            )
        hidden = self.initial_activation(self.initial_conv(x.float()))
        for block in self.residual_blocks:
            hidden = block(hidden)
        if hidden.shape[-1] != x.shape[-1]:
            raise RuntimeError("RBPNet trunk changed positional length")
        return hidden

    def _pool_measurement(
        self,
        hidden: torch.Tensor,
        measurement_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if measurement_mask is None:
            return hidden.mean(dim=-1)
        weights = torch.as_tensor(
            measurement_mask, device=hidden.device, dtype=hidden.dtype
        )
        expected = (hidden.shape[0], hidden.shape[-1])
        if weights.shape != expected:
            raise ValueError(
                f"measurement_mask shape {tuple(weights.shape)} does not match {expected}"
            )
        if torch.any(weights < 0):
            raise ValueError("measurement_mask weights must be non-negative")
        denominator = weights.sum(dim=-1, keepdim=True)
        if torch.any(denominator <= 0):
            raise ValueError("every measurement_mask row must have positive weight")
        return (hidden * weights.unsqueeze(1)).sum(dim=-1) / denominator

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        measurement_mask: torch.Tensor | None = None,
        profile_mask: torch.Tensor | None = None,
    ) -> RBPNetOutput:
        """Predict latent profiles, their IP mixture, and optional enrichment."""

        hidden = self.encode(sequence)
        target_logits = self.target_profile_head(hidden)
        control_logits = self.control_profile_head(hidden)
        if target_logits.shape != control_logits.shape or target_logits.shape[-1] != sequence.shape[-1]:
            raise RuntimeError("profile heads did not preserve positional shape")
        target_log_probs, target_probs = _masked_log_softmax(target_logits, profile_mask)
        control_log_probs, control_probs = _masked_log_softmax(control_logits, profile_mask)

        # Exclude legacy bundle padding from the global mixture summary when a
        # validity mask is available. Shift-to-fit bundles are all-valid and
        # therefore reduce to ordinary global average pooling.
        global_hidden = self._pool_measurement(hidden, profile_mask)
        mixing_logit = self.mixing_head(global_hidden).squeeze(-1)
        pi = torch.sigmoid(mixing_logit)
        log_pi = F.logsigmoid(mixing_logit).unsqueeze(-1)
        log_one_minus_pi = F.logsigmoid(-mixing_logit).unsqueeze(-1)
        ip_log_probs = torch.logaddexp(
            log_pi + target_log_probs,
            log_one_minus_pi + control_log_probs,
        )
        ip_probs = ip_log_probs.exp()
        if profile_mask is not None:
            valid = torch.as_tensor(profile_mask, device=sequence.device).bool()
            ip_log_probs = torch.where(valid, ip_log_probs, torch.zeros_like(ip_log_probs))
            ip_probs = torch.where(valid, ip_probs, torch.zeros_like(ip_probs))

        enrichment_logit = None
        if self.enrichment_head is not None:
            measurement_hidden = self._pool_measurement(hidden, measurement_mask)
            enrichment_logit = self.enrichment_head(measurement_hidden).squeeze(-1)

        return RBPNetOutput(
            target_logits=target_logits,
            control_logits=control_logits,
            target_log_probs=target_log_probs,
            control_log_probs=control_log_probs,
            target_probs=target_probs,
            control_probs=control_probs,
            mixing_logit=mixing_logit,
            pi=pi,
            ip_log_probs=ip_log_probs,
            ip_probs=ip_probs,
            enrichment_logit=enrichment_logit,
        )
