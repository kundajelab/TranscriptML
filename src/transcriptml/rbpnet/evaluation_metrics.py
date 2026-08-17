"""Numerically explicit metrics for structured RBPNet evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import pearsonr, spearmanr


PROFILE_METRIC_DEFINITIONS = {
    "multinomial_nll": (
        "Complete multinomial negative log likelihood, including the count "
        "combinatorial constant; lower is better."
    ),
    "multinomial_nll_without_constant": (
        "Multinomial cross-entropy term -sum(count * log probability), used "
        "only to reconstruct checkpoints trained with the optional count "
        "combinatorial constant disabled."
    ),
    "kl_per_read": (
        "(NLL_model - NLL_saturated) / observed profile reads, equal to "
        "KL(empirical || model) in natural-log units; zero is ideal."
    ),
    "jsd": (
        "Jensen-Shannon divergence between empirical and predicted normalized "
        "profiles in natural-log units; bounded by ln(2), zero is ideal."
    ),
    "information_gain_uniform_per_read": (
        "(LL_model - LL_uniform) / observed profile reads in natural-log units; "
        "larger is better."
    ),
    "information_gain_control_per_read": (
        "For pooled IP only, (LL_predicted_IP - LL_predicted_control) / observed "
        "IP reads in natural-log units; larger is better."
    ),
    "wasserstein_nt": (
        "One-dimensional earth-mover/Wasserstein-1 distance between empirical "
        "and predicted normalized profiles, in nucleotide units; zero is ideal."
    ),
}

ENRICHMENT_METRIC_DEFINITIONS = {
    "binomial_nll": (
        "Complete replicate-aware binomial negative log likelihood using "
        "logit(p_ij)=eta_i+log(L_IP,j/L_SM)."
    ),
    "binomial_nll_without_constant": (
        "Binomial cross-entropy term without log(N choose IP), used only to "
        "reconstruct checkpoints trained with that optional constant disabled."
    ),
    "information_gain_depth_null_per_read": (
        "(LL_model - LL_eta=0_depth_only) / (IP+SMInput) in natural-log units; "
        "larger is better."
    ),
    "empirical_eta": (
        "log((IP+c)/(SMInput+c))-log(L_IP/L_SM), used only for descriptive "
        "correlation and plotting, never as a likelihood target."
    ),
}


def _as_2d(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (N, L) or (L,)")
    return result


def profile_metrics(
    counts: np.ndarray,
    probabilities: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    control_probabilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Calculate complete-likelihood and normalized profile-shape metrics.

    Rows with zero observed counts retain their count and valid-position count
    but receive ``NaN`` for empirical-profile metrics. Probabilities are
    normalized over valid positions; counts outside the mask are rejected.
    Natural logarithms are used throughout.
    """

    observed = _as_2d(np.asarray(counts, dtype=np.float64), "counts")
    predicted = _as_2d(
        np.asarray(probabilities, dtype=np.float64), "probabilities"
    )
    if observed.shape != predicted.shape:
        raise ValueError("counts and probabilities must have matching shapes")
    if np.any(observed < 0) or not np.all(np.isfinite(observed)):
        raise ValueError("profile counts must be finite and non-negative")
    if np.any(predicted < 0) or not np.all(np.isfinite(predicted)):
        raise ValueError("profile probabilities must be finite and non-negative")
    if valid_mask is None:
        valid = np.ones(observed.shape, dtype=bool)
    else:
        valid = _as_2d(np.asarray(valid_mask, dtype=bool), "valid_mask")
        if valid.shape != observed.shape:
            raise ValueError("valid_mask must match profile shape")
    if np.any((~valid) & (observed != 0)):
        raise ValueError("profile counts occur outside the validity mask")
    if np.any(valid.sum(axis=1) == 0):
        raise ValueError("every profile requires at least one valid position")

    control = None
    if control_probabilities is not None:
        control = _as_2d(
            np.asarray(control_probabilities, dtype=np.float64),
            "control_probabilities",
        )
        if control.shape != observed.shape:
            raise ValueError("control_probabilities must match profile shape")
        if np.any(control < 0) or not np.all(np.isfinite(control)):
            raise ValueError(
                "control profile probabilities must be finite and non-negative"
            )

    n_rows = observed.shape[0]
    names = (
        "multinomial_nll",
        "multinomial_nll_without_constant",
        "saturated_nll",
        "uniform_nll",
        "kl_per_read",
        "jsd",
        "information_gain_uniform_per_read",
        "wasserstein_nt",
    )
    result = {
        "count": observed.sum(axis=1),
        "valid_positions": valid.sum(axis=1).astype(np.int64),
        **{name: np.full(n_rows, np.nan, dtype=np.float64) for name in names},
    }
    if control is not None:
        result["information_gain_control_per_read"] = np.full(
            n_rows, np.nan, dtype=np.float64
        )

    for index in range(n_rows):
        mask = valid[index]
        row_counts = observed[index, mask]
        total = float(row_counts.sum())
        if total <= 0:
            continue
        model = predicted[index, mask]
        model_sum = float(model.sum())
        if model_sum <= 0:
            raise ValueError("predicted profile has zero mass on valid positions")
        model = model / model_sum
        empirical = row_counts / total
        constant = float(
            gammaln(total + 1.0) - np.sum(gammaln(row_counts + 1.0))
        )

        positive = row_counts > 0
        with np.errstate(divide="ignore"):
            model_log_likelihood = constant + float(
                np.sum(row_counts[positive] * np.log(model[positive]))
            )
        saturated_log_likelihood = constant + float(
            np.sum(row_counts[positive] * np.log(empirical[positive]))
        )
        uniform_log_likelihood = constant - total * math.log(len(row_counts))
        mixture = 0.5 * (empirical + model)
        empirical_kl_mixture = float(
            np.sum(empirical[positive] * np.log(empirical[positive] / mixture[positive]))
        )
        model_positive = model > 0
        model_kl_mixture = float(
            np.sum(model[model_positive] * np.log(model[model_positive] / mixture[model_positive]))
        )

        result["multinomial_nll"][index] = -model_log_likelihood
        result["multinomial_nll_without_constant"][index] = -(
            model_log_likelihood - constant
        )
        result["saturated_nll"][index] = -saturated_log_likelihood
        result["uniform_nll"][index] = -uniform_log_likelihood
        result["kl_per_read"][index] = (
            saturated_log_likelihood - model_log_likelihood
        ) / total
        result["jsd"][index] = 0.5 * (
            empirical_kl_mixture + model_kl_mixture
        )
        result["information_gain_uniform_per_read"][index] = (
            model_log_likelihood - uniform_log_likelihood
        ) / total
        result["wasserstein_nt"][index] = float(
            np.abs(np.cumsum(empirical) - np.cumsum(model)).sum()
        )

        if control is not None:
            control_row = control[index, mask]
            control_sum = float(control_row.sum())
            if control_sum <= 0:
                raise ValueError(
                    "control profile has zero mass on valid positions"
                )
            control_row = control_row / control_sum
            with np.errstate(divide="ignore"):
                control_log_likelihood = constant + float(
                    np.sum(row_counts[positive] * np.log(control_row[positive]))
                )
            result["information_gain_control_per_read"][index] = (
                model_log_likelihood - control_log_likelihood
            ) / total
    return result


def enrichment_metrics(
    eta: np.ndarray,
    ip_counts: np.ndarray,
    sminput_counts: np.ndarray,
    depth_offsets: np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> dict[str, np.ndarray]:
    """Calculate replicate-aware binomial metrics and descriptive enrichment."""

    eta = np.asarray(eta, dtype=np.float64).reshape(-1)
    ip = np.asarray(ip_counts, dtype=np.float64)
    if ip.ndim == 1:
        ip = ip[:, None]
    sm = np.asarray(sminput_counts, dtype=np.float64).reshape(-1)
    offsets = np.asarray(depth_offsets, dtype=np.float64)
    if ip.ndim != 2 or ip.shape[0] != eta.size or sm.shape != eta.shape:
        raise ValueError("eta, IP counts, and SMInput counts are not aligned")
    if offsets.ndim == 1:
        if offsets.shape[0] != ip.shape[1]:
            raise ValueError("depth_offsets must have one value per IP replicate")
        offsets = np.broadcast_to(offsets[None, :], ip.shape)
    elif offsets.shape != ip.shape:
        raise ValueError("depth_offsets must have shape (R,) or (N, R)")
    if pseudocount <= 0:
        raise ValueError("enrichment pseudocount must be positive")
    if np.any(ip < 0) or np.any(sm < 0):
        raise ValueError("enrichment counts must be non-negative")

    failures = np.broadcast_to(sm[:, None], ip.shape)
    total = ip + failures
    logits = eta[:, None] + offsets
    predicted = expit(logits)
    depth_only = expit(offsets)
    log_choose = gammaln(total + 1.0) - gammaln(ip + 1.0) - gammaln(
        failures + 1.0
    )

    def nll_for_logits(value: np.ndarray) -> np.ndarray:
        return total * np.logaddexp(0.0, value) - ip * value - log_choose

    model_nll_without_constant = (
        total * np.logaddexp(0.0, logits) - ip * logits
    )
    model_nll = nll_for_logits(logits)
    null_nll = nll_for_logits(offsets)
    informative = total > 0
    model_nll = np.where(informative, model_nll, np.nan)
    model_nll_without_constant = np.where(
        informative, model_nll_without_constant, np.nan
    )
    null_nll = np.where(informative, null_nll, np.nan)
    information_gain = np.divide(
        null_nll - model_nll,
        total,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=informative,
    )
    observed_fraction = np.divide(
        ip,
        total,
        out=np.full(ip.shape, np.nan, dtype=np.float64),
        where=informative,
    )
    empirical_eta = np.where(
        informative,
        np.log((ip + pseudocount) / (failures + pseudocount)) - offsets,
        np.nan,
    )
    return {
        "count": total,
        "predicted_probability": predicted,
        "observed_fraction": observed_fraction,
        "binomial_nll": model_nll,
        "binomial_nll_without_constant": model_nll_without_constant,
        "depth_null_nll": null_nll,
        "information_gain_depth_null_per_read": information_gain,
        "empirical_eta": empirical_eta,
    }


def replicate_ceiling_metrics(
    replicate_profiles: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compare each IP replicate with the pooled profile of all other replicates."""

    profiles = np.asarray(replicate_profiles, dtype=np.float64)
    if profiles.ndim != 3:
        raise ValueError("replicate_profiles must have shape (N, R, L)")
    n_examples, n_replicates, length = profiles.shape
    if valid_mask is None:
        valid = np.ones((n_examples, length), dtype=bool)
    else:
        valid = _as_2d(np.asarray(valid_mask, dtype=bool), "valid_mask")
        if valid.shape != (n_examples, length):
            raise ValueError("valid_mask must have shape (N, L)")
    if n_replicates < 2:
        empty = np.empty((n_examples, 0), dtype=np.float64)
        return {"count": empty, "jsd": empty, "wasserstein_nt": empty}

    counts = profiles.sum(axis=2)
    jsd = np.full((n_examples, n_replicates), np.nan, dtype=np.float64)
    wasserstein = np.full_like(jsd, np.nan)
    pooled = profiles.sum(axis=1)
    for replicate in range(n_replicates):
        observed = profiles[:, replicate, :]
        leave_one_out = pooled - observed
        leave_one_out_counts = leave_one_out.sum(axis=1)
        informative = (counts[:, replicate] > 0) & (leave_one_out_counts > 0)
        if np.any(informative):
            metrics = profile_metrics(
                observed[informative],
                leave_one_out[informative],
                valid_mask=valid[informative],
            )
            jsd[informative, replicate] = metrics["jsd"]
            wasserstein[informative, replicate] = metrics["wasserstein_nt"]
    return {"count": counts, "jsd": jsd, "wasserstein_nt": wasserstein}


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Return finite-pair Pearson/Spearman correlations without warnings."""

    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    result: dict[str, float | int] = {"n": int(left.size)}
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        result.update({"pearson": float("nan"), "spearman": float("nan")})
        return result
    result.update(
        {
            "pearson": float(pearsonr(left, right).statistic),
            "spearman": float(spearmanr(left, right).statistic),
        }
    )
    return result


def calibration_rows(
    predicted_probability: np.ndarray,
    ip_counts: np.ndarray,
    total_counts: np.ndarray,
    replicate_names: Sequence[str],
    *,
    n_bins: int = 10,
) -> list[dict[str, object]]:
    """Build read-weighted fixed-width predicted-probability calibration bins."""

    predicted = np.asarray(predicted_probability, dtype=np.float64)
    successes = np.asarray(ip_counts, dtype=np.float64)
    totals = np.asarray(total_counts, dtype=np.float64)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if predicted.shape != successes.shape or predicted.shape != totals.shape:
        raise ValueError("calibration arrays must have matching (N, R) shapes")
    if predicted.shape[1] != len(replicate_names):
        raise ValueError("replicate_names does not match calibration arrays")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    rows: list[dict[str, object]] = []
    for replicate, name in enumerate(replicate_names):
        values = predicted[:, replicate]
        valid = (
            np.isfinite(values)
            & (values >= 0)
            & (values <= 1)
            & (totals[:, replicate] > 0)
        )
        indices = np.zeros(values.shape, dtype=np.int64)
        indices[valid] = np.minimum(
            (values[valid] * n_bins).astype(np.int64), n_bins - 1
        )
        for bin_index in range(n_bins):
            selected = valid & (indices == bin_index)
            if not np.any(selected):
                continue
            weight = float(totals[selected, replicate].sum())
            rows.append(
                {
                    "replicate": str(name),
                    "bin": bin_index,
                    "bin_left": bin_index / n_bins,
                    "bin_right": (bin_index + 1) / n_bins,
                    "n_observations": int(selected.sum()),
                    "total_reads": int(round(weight)),
                    "predicted_bin": float(
                        np.sum(totals[selected, replicate] * values[selected])
                        / weight
                    ),
                    "observed_bin": float(
                        successes[selected, replicate].sum() / weight
                    ),
                }
            )
    return rows


def aggregate_observations(
    values: np.ndarray,
    *,
    locus_ids: Sequence[object],
    gene_ids: Sequence[object],
    read_weights: np.ndarray | None = None,
    micro_numerators: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    """Calculate locus-macro, gene-macro, and read-micro summaries.

    Replicate observations sharing a locus are averaged before locus- and
    gene-macro aggregation. Read micro is a read-weighted mean, or equivalently
    ``sum(micro_numerators)/sum(read_weights)`` when explicit numerators are
    supplied (for example complete NLL rather than NLL times read count).
    """

    metric = np.asarray(values, dtype=np.float64).reshape(-1)
    loci = np.asarray(locus_ids, dtype=object).reshape(-1)
    genes = np.asarray(gene_ids, dtype=object).reshape(-1)
    if metric.shape != loci.shape or metric.shape != genes.shape:
        raise ValueError("values, locus_ids, and gene_ids must be aligned")
    weights = (
        np.ones(metric.shape, dtype=np.float64)
        if read_weights is None
        else np.asarray(read_weights, dtype=np.float64).reshape(-1)
    )
    if weights.shape != metric.shape:
        raise ValueError("read_weights must align with values")
    numerators = (
        metric * weights
        if micro_numerators is None
        else np.asarray(micro_numerators, dtype=np.float64).reshape(-1)
    )
    if numerators.shape != metric.shape:
        raise ValueError("micro_numerators must align with values")
    keep = (
        np.isfinite(metric)
        & np.isfinite(weights)
        & np.isfinite(numerators)
        & (weights > 0)
    )
    metric = metric[keep]
    weights = weights[keep]
    numerators = numerators[keep]
    loci = loci[keep]
    genes = genes[keep]
    if metric.size == 0:
        empty = {
            "value": float("nan"),
            "n_observations": 0,
            "n_loci": 0,
            "n_genes": 0,
            "total_reads": 0.0,
        }
        return {
            "locus_macro": dict(empty),
            "gene_macro": dict(empty),
            "read_micro": dict(empty),
        }

    locus_values: dict[object, list[float]] = defaultdict(list)
    locus_gene: dict[object, object] = {}
    for value, locus, gene in zip(metric, loci, genes):
        locus_values[locus].append(float(value))
        locus_gene.setdefault(locus, gene)
    locus_means = {key: float(np.mean(value)) for key, value in locus_values.items()}
    gene_values: dict[object, list[float]] = defaultdict(list)
    for locus, value in locus_means.items():
        gene_values[locus_gene[locus]].append(value)
    gene_means = [float(np.mean(value)) for value in gene_values.values()]
    common = {
        "n_observations": int(metric.size),
        "n_loci": len(locus_means),
        "n_genes": len(gene_values),
        "total_reads": float(weights.sum()),
    }
    return {
        "locus_macro": {
            **common,
            "value": float(np.mean(list(locus_means.values()))),
        },
        "gene_macro": {**common, "value": float(np.mean(gene_means))},
        "read_micro": {
            **common,
            "value": float(numerators.sum() / weights.sum()),
        },
    }


def select_representative_examples(
    metric: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    seed: int = 123,
    per_tier: int = 3,
) -> list[dict[str, object]]:
    """Reproducibly sample good/middle/poor examples from rank tertiles."""

    values = np.asarray(metric, dtype=np.float64).reshape(-1)
    allowed = np.isfinite(values)
    if eligible is not None:
        requested = np.asarray(eligible, dtype=bool).reshape(-1)
        if requested.shape != values.shape:
            raise ValueError("eligible must align with metric")
        allowed &= requested
    if per_tier < 0:
        raise ValueError("per_tier must be non-negative")
    ordered = np.flatnonzero(allowed)
    ordered = ordered[np.lexsort((ordered, values[ordered]))]
    tiers = np.array_split(ordered, 3)
    labels = ("good", "intermediate", "poor")
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, object]] = []
    for tier_index, (label, candidates) in enumerate(zip(labels, tiers)):
        size = min(int(per_tier), len(candidates))
        if size == 0:
            continue
        chosen = np.sort(rng.choice(candidates, size=size, replace=False))
        for index in chosen:
            selected.append(
                {
                    "index": int(index),
                    "tier": label,
                    "quantile_left": tier_index / 3,
                    "quantile_right": (tier_index + 1) / 3,
                    "metric_value": float(values[index]),
                }
            )
    return selected
