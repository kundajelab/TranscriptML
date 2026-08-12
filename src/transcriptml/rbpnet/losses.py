"""Stable structured likelihoods for TranscriptML RBPNet models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from transcriptml.models.rbpnet import RBPNetOutput
from transcriptml.rbpnet.dataset import RBPNetBatch


@dataclass(frozen=True)
class ReducedLikelihood:
    """A reduced likelihood and exact aggregation terms."""

    loss: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    per_observation: torch.Tensor
    valid: torch.Tensor


@dataclass
class RBPNetLossConfig:
    """Weights and reporting choices for the structured RBPNet objective."""

    name: str = "rbpnet"
    lambda_ip_profile: float = 1.0
    lambda_sm_profile: float = 1.0
    lambda_enrichment: float = 1.0
    include_multinomial_constant: bool = True
    include_binomial_constant: bool = True

    @classmethod
    def from_config(cls, config: str | Mapping[str, object] | None) -> "RBPNetLossConfig":
        if config is None:
            return cls()
        if isinstance(config, str):
            values: dict[str, object] = {"name": config}
        else:
            values = dict(config)
        name = str(values.pop("name", "rbpnet")).strip().lower()
        values.pop("enrichment_enabled", None)
        values.pop("effective_lambda_enrichment", None)
        if name not in {"rbpnet", "rbpnet_profile", "rbpnet_profile_enrichment"}:
            raise ValueError(
                "RBPNet training requires loss.name='rbpnet', not " + repr(name)
            )
        result = cls(name="rbpnet", **values)
        for field_name in (
            "lambda_ip_profile",
            "lambda_sm_profile",
            "lambda_enrichment",
        ):
            if float(getattr(result, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if result.lambda_ip_profile == result.lambda_sm_profile == result.lambda_enrichment == 0:
            raise ValueError("at least one RBPNet loss weight must be positive")
        return result

    def to_dict(self, *, enrichment_enabled: bool | None = None) -> dict[str, object]:
        values = asdict(self)
        if enrichment_enabled is not None:
            values["enrichment_enabled"] = bool(enrichment_enabled)
            values["effective_lambda_enrichment"] = (
                float(self.lambda_enrichment) if enrichment_enabled else 0.0
            )
        return values


@dataclass(frozen=True)
class RBPNetLossOutput:
    """Total differentiable loss plus independently aggregatable components."""

    loss: torch.Tensor
    components: Mapping[str, torch.Tensor]
    numerators: Mapping[str, torch.Tensor]
    denominators: Mapping[str, torch.Tensor]


def _reduce_valid(nll: torch.Tensor, valid: torch.Tensor) -> ReducedLikelihood:
    valid = valid.bool()
    numerator = torch.where(valid, nll, torch.zeros_like(nll)).sum()
    denominator = valid.sum().to(dtype=nll.dtype)
    loss = numerator / denominator.clamp_min(1.0)
    return ReducedLikelihood(loss, numerator, denominator, nll, valid)


def multinomial_nll(
    log_probs: torch.Tensor,
    counts: torch.Tensor,
    *,
    valid_positions: torch.Tensor | None = None,
    include_constant: bool = True,
) -> ReducedLikelihood:
    """Mean multinomial NLL over loci with nonzero profile totals.

    Zero-total profiles contain no positional information and are excluded from
    the mean. With ``include_constant=True`` (the default), this is the complete
    multinomial NLL, including the ``lgamma`` combinatorial term.
    """

    log_probs = log_probs.float()
    counts = counts.to(device=log_probs.device, dtype=log_probs.dtype)
    if log_probs.ndim != 2 or counts.shape != log_probs.shape:
        raise ValueError("multinomial log_probs and counts must have matching (B, L) shapes")
    if torch.any(counts < 0) or not torch.all(torch.isfinite(counts)):
        raise ValueError("multinomial counts must be finite and non-negative")
    if valid_positions is not None:
        mask = valid_positions.to(device=log_probs.device).bool()
        if mask.shape != counts.shape:
            raise ValueError("valid_positions must match multinomial count shape")
        if torch.any((~mask) & (counts != 0)):
            raise ValueError("multinomial counts occur outside the valid profile mask")
    total = counts.sum(dim=-1)
    safe_terms = torch.where(counts > 0, counts * log_probs, torch.zeros_like(counts))
    log_likelihood = safe_terms.sum(dim=-1)
    if include_constant:
        log_likelihood = log_likelihood + torch.lgamma(total + 1) - torch.lgamma(
            counts + 1
        ).sum(dim=-1)
    nll = -log_likelihood
    return _reduce_valid(nll, total > 0)


def replicate_binomial_nll(
    eta: torch.Tensor,
    ip_counts: torch.Tensor,
    sminput_counts: torch.Tensor,
    depth_offsets: torch.Tensor,
    *,
    include_constant: bool = True,
) -> ReducedLikelihood:
    """Binomial NLL over valid locus-replicate observations.

    ``eta`` is one sequence-derived log enrichment per locus. Known effective
    library sizes enter only through ``depth_offsets = log(L_IP/L_SM)``.
    """

    eta = eta.float().reshape(-1)
    ip_counts = ip_counts.to(device=eta.device, dtype=eta.dtype)
    sminput_counts = sminput_counts.to(device=eta.device, dtype=eta.dtype).reshape(-1)
    depth_offsets = depth_offsets.to(device=eta.device, dtype=eta.dtype)
    if ip_counts.ndim != 2 or ip_counts.shape[0] != eta.shape[0]:
        raise ValueError("ip_counts must have shape (B, R) aligned to eta")
    if sminput_counts.shape != eta.shape:
        raise ValueError("sminput_counts must have shape (B,)")
    if depth_offsets.ndim == 1:
        if depth_offsets.shape[0] != ip_counts.shape[1]:
            raise ValueError("one-dimensional depth_offsets must have shape (R,)")
        offsets = depth_offsets.unsqueeze(0).expand_as(ip_counts)
    elif depth_offsets.shape == ip_counts.shape:
        offsets = depth_offsets
    else:
        raise ValueError("depth_offsets must have shape (R,) or (B, R)")
    if torch.any(ip_counts < 0) or torch.any(sminput_counts < 0):
        raise ValueError("IP and SMInput measurement counts must be non-negative")
    failures = sminput_counts.unsqueeze(-1).expand_as(ip_counts)
    total = ip_counts + failures
    logits = eta.unsqueeze(-1) + offsets
    # N * softplus(logit) - k * logit is the logits-based binomial
    # cross-entropy. Subtract log(N choose k) for the complete NLL.
    nll = total * F.softplus(logits) - ip_counts * logits
    if include_constant:
        log_choose = (
            torch.lgamma(total + 1)
            - torch.lgamma(ip_counts + 1)
            - torch.lgamma(failures + 1)
        )
        nll = nll - log_choose
    return _reduce_valid(nll, total > 0)


class RBPNetObjective(nn.Module):
    """Target/control profile objective with an optional independent eta head."""

    def __init__(
        self,
        config: RBPNetLossConfig | Mapping[str, object] | str | None = None,
        *,
        enrichment_enabled: bool,
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, RBPNetLossConfig)
            else RBPNetLossConfig.from_config(config)
        )
        self.enrichment_enabled = bool(enrichment_enabled)
        effective_weight = (
            float(self.config.lambda_ip_profile)
            + float(self.config.lambda_sm_profile)
            + (
                float(self.config.lambda_enrichment)
                if self.enrichment_enabled
                else 0.0
            )
        )
        if effective_weight == 0:
            raise ValueError("enabled RBPNet loss components cannot all have zero weight")

    def forward(self, output: RBPNetOutput, batch: RBPNetBatch) -> RBPNetLossOutput:
        pooled_from_replicates = batch.individual_ip_profiles.sum(dim=1)
        if not torch.equal(pooled_from_replicates, batch.pooled_ip_profile):
            raise ValueError("pooled IP profile does not equal the sum over replicate profiles")
        ip = multinomial_nll(
            output.ip_log_probs,
            batch.pooled_ip_profile,
            valid_positions=batch.profile_valid_mask,
            include_constant=self.config.include_multinomial_constant,
        )
        sm = multinomial_nll(
            output.control_log_probs,
            batch.sminput_profile,
            valid_positions=batch.profile_valid_mask,
            include_constant=self.config.include_multinomial_constant,
        )
        zero = output.ip_log_probs.sum() * 0.0
        if self.enrichment_enabled:
            if output.enrichment_logit is None:
                raise ValueError("enrichment-enabled objective requires model enrichment_logit")
            enrichment = replicate_binomial_nll(
                output.enrichment_logit,
                batch.ip_measurement_counts,
                batch.sminput_measurement_counts,
                batch.depth_offsets,
                include_constant=self.config.include_binomial_constant,
            )
        else:
            enrichment = ReducedLikelihood(
                loss=zero,
                numerator=zero,
                denominator=zero.detach(),
                per_observation=zero.reshape(1),
                valid=torch.zeros(1, dtype=torch.bool, device=zero.device),
            )
        total = (
            float(self.config.lambda_ip_profile) * ip.loss
            + float(self.config.lambda_sm_profile) * sm.loss
            + (
                float(self.config.lambda_enrichment) * enrichment.loss
                if self.enrichment_enabled
                else zero
            )
        )
        return RBPNetLossOutput(
            loss=total,
            components={
                "ip_profile_loss": ip.loss,
                "sm_profile_loss": sm.loss,
                "enrichment_loss": enrichment.loss,
            },
            numerators={
                "ip_profile_loss": ip.numerator,
                "sm_profile_loss": sm.numerator,
                "enrichment_loss": enrichment.numerator,
            },
            denominators={
                "ip_profile_loss": ip.denominator,
                "sm_profile_loss": sm.denominator,
                "enrichment_loss": enrichment.denominator,
            },
        )
