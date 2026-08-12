"""Structured, checkpoint-split-aware evaluation reports for RBPNet."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from transcriptml.data.bundle import DatasetBundle
from transcriptml.devices import resolve_device
from transcriptml.models.rbpnet import RBPNet
from transcriptml.progress import ProgressReporter, log_progress
from transcriptml.rbpnet.dataset import RBPNetDataset, collate_rbpnet
from transcriptml.rbpnet.evaluation_metrics import (
    ENRICHMENT_METRIC_DEFINITIONS,
    PROFILE_METRIC_DEFINITIONS,
    aggregate_observations,
    calibration_rows,
    enrichment_metrics,
    profile_metrics,
    replicate_ceiling_metrics,
    safe_correlations,
    select_representative_examples,
)
from transcriptml.rbpnet.losses import RBPNetLossConfig


PROFILE_TRACKS = ("pooled_ip", "sminput")
PROFILE_METRICS = (
    "multinomial_nll",
    "kl_per_read",
    "jsd",
    "information_gain_uniform_per_read",
    "wasserstein_nt",
)


def resolve_rbpnet_checkpoint_indices(
    checkpoint: Mapping[str, object],
    *,
    split: str | None,
    n_examples: int,
) -> tuple[str, list[int]]:
    """Resolve a named RBPNet split exclusively from checkpoint artifacts."""

    requested = "test" if split is None else str(split).strip().lower()
    if requested not in {"train", "val", "test", "all"}:
        raise ValueError("RBPNet --split must be one of: train, val, test, all")
    raw = checkpoint.get("splits")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "RBPNet checkpoint has no recorded training splits; evaluation will not "
            "fall back to bundle.splits"
        )
    normalized: dict[str, list[int]] = {}
    owners: dict[int, str] = {}
    for name in ("train", "val", "test"):
        values = raw.get(name)
        if values is None or isinstance(values, (str, bytes)):
            raise ValueError(f"RBPNet checkpoint lacks recorded {name!r} indices")
        indices = [int(value) for value in values]
        if len(indices) != len(set(indices)):
            raise ValueError(f"RBPNet checkpoint {name!r} split contains duplicates")
        if any(index < 0 or index >= int(n_examples) for index in indices):
            raise ValueError(
                f"RBPNet checkpoint {name!r} split contains an out-of-range index"
            )
        for index in indices:
            previous = owners.setdefault(index, name)
            if previous != name:
                raise ValueError(
                    f"RBPNet checkpoint index {index} occurs in both {previous} and {name}"
                )
        normalized[name] = indices
    indices = (
        sorted(owners)
        if requested == "all"
        else list(normalized[requested])
    )
    if not indices:
        raise ValueError(f"RBPNet checkpoint split {requested!r} is empty")
    return requested, indices


def _loader(
    dataset: RBPNetDataset,
    indices: Sequence[int],
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rbpnet,
    )


def _mean_finite(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _weighted_row_mean(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerators = np.where(valid, values * weights, 0.0).sum(axis=1)
    denominators = np.where(valid, weights, 0.0).sum(axis=1)
    means = np.divide(
        numerators,
        denominators,
        out=np.full(denominators.shape, np.nan, dtype=np.float64),
        where=denominators > 0,
    )
    return means, numerators, denominators


def _metadata_value(row: Mapping[str, object], name: str) -> object | None:
    value = row.get(name)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _base_example_row(
    bundle: DatasetBundle,
    index: int,
    evaluation_row: int,
    split: str,
) -> dict[str, object]:
    metadata = bundle.metadata[index] if bundle.metadata is not None else {}
    row: dict[str, object] = {
        "evaluation_row": int(evaluation_row),
        "index": int(index),
        "example_id": str(bundle.ids[index]),
        "evaluated_split": split,
    }
    for name in (
        "gene_id",
        "transcript_id",
        "chromosome",
        "strand",
        "coordinate_space",
        "locus_length",
        "transcript_anchor",
        "selection_start",
        "selection_end",
        "selection_state",
        "selection_strategy",
        "replicate_id",
        "region_type",
        "sequence_materialized_start",
        "sequence_materialized_end",
        "profile_materialized_start",
        "profile_materialized_end",
        "group_gene_id",
        "group_transcript_id",
        "group_chromosome",
    ):
        value = _metadata_value(metadata, name)
        if value is not None:
            row[name] = value
    return row


def _profile_columns(
    row: dict[str, object],
    prefix: str,
    metrics: Mapping[str, np.ndarray],
    position: int,
) -> None:
    row[f"{prefix}_profile_count"] = int(metrics["count"][position])
    row[f"{prefix}_valid_positions"] = int(metrics["valid_positions"][position])
    for name in (
        "multinomial_nll",
        "multinomial_nll_without_constant",
        "saturated_nll",
        "uniform_nll",
        "kl_per_read",
        "jsd",
        "information_gain_uniform_per_read",
        "wasserstein_nt",
        "information_gain_control_per_read",
    ):
        if name in metrics:
            row[f"{prefix}_{name}"] = float(metrics[name][position])


def _depth_bin(value: float) -> str:
    count = int(value)
    if count <= 0:
        return "0"
    for upper, label in (
        (2, "1-2"),
        (5, "3-5"),
        (10, "6-10"),
        (20, "11-20"),
        (50, "21-50"),
        (100, "51-100"),
    ):
        if count <= upper:
            return label
    return ">100"


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _metric_unit(metric: str, aggregation: str) -> str:
    if metric == "wasserstein_nt":
        return "nt"
    if metric == "multinomial_nll":
        return "nats/read" if aggregation == "read_micro" else "nats/locus"
    if metric == "binomial_nll":
        return "nats/read" if aggregation == "read_micro" else "nats/observation"
    return "nats/read" if "information_gain" in metric or "kl_per_read" in metric else "nats"


def _stratified_metric_rows(
    examples: Sequence[Mapping[str, object]],
    replicate_names: Sequence[str],
) -> list[dict[str, object]]:
    """Create long-form overall and stratified metric summaries."""

    rows = list(examples)
    result: list[dict[str, object]] = []
    if not rows:
        return result
    locus_ids = np.asarray([row["example_id"] for row in rows], dtype=object)
    gene_ids = np.asarray(
        [row.get("gene_id", row.get("transcript_id", row["example_id"])) for row in rows],
        dtype=object,
    )

    base_strata: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(rows), dtype=bool))
    ]
    for field, dimension in (
        ("selection_state", "selection_state"),
        ("region_type", "region_type"),
        ("chromosome", "chromosome"),
    ):
        values = np.asarray([row.get(field) for row in rows], dtype=object)
        for value in sorted({str(value) for value in values if value not in {None, ""}}):
            base_strata.append((dimension, value, values == value))

    for track in PROFILE_TRACKS:
        track_strata = list(base_strata)
        depth = np.asarray([row[f"{track}_profile_count"] for row in rows], dtype=float)
        depth_labels = np.asarray([_depth_bin(value) for value in depth], dtype=object)
        for label in ("0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"):
            if np.any(depth_labels == label):
                track_strata.append((f"{track}_read_depth", label, depth_labels == label))

        metrics = list(PROFILE_METRICS)
        if track == "pooled_ip":
            metrics.append("information_gain_control_per_read")
        for dimension, stratum, selected in track_strata:
            for metric in metrics:
                column = f"{track}_{metric}"
                values = np.asarray([row.get(column, np.nan) for row in rows], dtype=float)
                nll = metric == "multinomial_nll"
                summary = aggregate_observations(
                    values[selected],
                    locus_ids=locus_ids[selected],
                    gene_ids=gene_ids[selected],
                    read_weights=depth[selected],
                    micro_numerators=values[selected] if nll else None,
                )
                for aggregation, details in summary.items():
                    result.append(
                        {
                            "dimension": dimension,
                            "stratum": stratum,
                            "track": track,
                            "metric": metric,
                            "aggregation": aggregation,
                            "value": details["value"],
                            "unit": _metric_unit(metric, aggregation),
                            "n_observations": details["n_observations"],
                            "n_loci": details["n_loci"],
                            "n_genes": details["n_genes"],
                            "total_reads": details["total_reads"],
                        }
                    )

    if "enrichment_binomial_nll" in rows[0]:
        n_replicates = len(replicate_names)
        counts = np.asarray(
            [
                [row.get(f"replicate_{name}_enrichment_count", 0) for name in replicate_names]
                for row in rows
            ],
            dtype=float,
        ).reshape(-1)
        enrichment_loci = np.repeat(locus_ids, n_replicates)
        enrichment_genes = np.repeat(gene_ids, n_replicates)
        enrichment_strata = [
            (
                dimension,
                stratum,
                np.repeat(selected[:, None], n_replicates, axis=1).reshape(-1),
            )
            for dimension, stratum, selected in base_strata
        ]
        depth_labels = np.asarray([_depth_bin(value) for value in counts], dtype=object)
        for label in ("0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"):
            if np.any(depth_labels == label):
                enrichment_strata.append(
                    ("enrichment_read_depth", label, depth_labels == label)
                )
        for dimension, stratum, selected in enrichment_strata:
            for metric in (
                "binomial_nll",
                "information_gain_depth_null_per_read",
            ):
                values = np.asarray(
                    [
                        row.get(f"replicate_{name}_{metric}", np.nan)
                        for row in rows
                        for name in replicate_names
                    ],
                    dtype=float,
                )
                summary = aggregate_observations(
                    values[selected],
                    locus_ids=enrichment_loci[selected],
                    gene_ids=enrichment_genes[selected],
                    read_weights=counts[selected],
                    micro_numerators=values[selected] if metric == "binomial_nll" else None,
                )
                for aggregation, details in summary.items():
                    result.append(
                        {
                            "dimension": dimension,
                            "stratum": stratum,
                            "track": "enrichment",
                            "metric": metric,
                            "aggregation": aggregation,
                            "value": details["value"],
                            "unit": _metric_unit(metric, aggregation),
                            "n_observations": details["n_observations"],
                            "n_loci": details["n_loci"],
                            "n_genes": details["n_genes"],
                            "total_reads": details["total_reads"],
                        }
                    )

    for metric in ("jsd", "wasserstein_nt"):
        if not replicate_names or not any(
            f"replicate_{name}_ceiling_{metric}" in rows[0]
            for name in replicate_names
        ):
            continue
        values = np.asarray(
            [
                row.get(f"replicate_{name}_ceiling_{metric}", np.nan)
                for row in rows
                for name in replicate_names
            ],
            dtype=float,
        )
        weights = np.asarray(
            [
                row.get(f"replicate_{name}_profile_count", 0)
                for row in rows
                for name in replicate_names
            ],
            dtype=float,
        )
        summary = aggregate_observations(
            values,
            locus_ids=np.repeat(locus_ids, len(replicate_names)),
            gene_ids=np.repeat(gene_ids, len(replicate_names)),
            read_weights=weights,
        )
        for aggregation, details in summary.items():
            result.append(
                {
                    "dimension": "overall",
                    "stratum": "all",
                    "track": "replicate_ceiling",
                    "metric": metric,
                    "aggregation": aggregation,
                    "value": details["value"],
                    "unit": _metric_unit(metric, aggregation),
                    "n_observations": details["n_observations"],
                    "n_loci": details["n_loci"],
                    "n_genes": details["n_genes"],
                    "total_reads": details["total_reads"],
                }
            )
    return result


def _calibration_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("row_type", pa.string()),
            pa.field("replicate", pa.string()),
            pa.field("evaluation_row", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("example_id", pa.string()),
            pa.field("eta", pa.float64()),
            pa.field("empirical_eta", pa.float64()),
            pa.field("predicted_probability", pa.float64()),
            pa.field("observed_fraction", pa.float64()),
            pa.field("ip_count", pa.int64()),
            pa.field("sminput_count", pa.int64()),
            pa.field("total_reads", pa.int64()),
            pa.field("bin", pa.int64()),
            pa.field("bin_left", pa.float64()),
            pa.field("bin_right", pa.float64()),
            pa.field("n_observations", pa.int64()),
            pa.field("predicted_bin", pa.float64()),
            pa.field("observed_bin", pa.float64()),
        ]
    )


def _write_table(rows: Sequence[Mapping[str, object]], path: Path, schema: pa.Schema | None = None) -> None:
    table = (
        pa.Table.from_pylist(list(rows), schema=schema)
        if rows or schema is not None
        else pa.table({})
    )
    pq.write_table(table, path, compression="zstd")


def evaluate_rbpnet_report(
    model: RBPNet,
    checkpoint: Mapping[str, object],
    bundle: DatasetBundle,
    output_dir: str | Path,
    *,
    split: str | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    save_profiles: bool = False,
    calibration_bins: int = 10,
    enrichment_pseudocount: float = 0.5,
    representative_seed: int = 123,
    representative_per_tier: int = 3,
    representative_min_profile_count: int = 10,
    checkpoint_path: str | Path | None = None,
    progress: bool = True,
) -> dict[str, object]:
    """Create a complete deterministic RBPNet evaluation report directory."""

    if bundle.config.get("bundle_format") != "transcriptml-rbpnet-bundle":
        raise ValueError("RBPNet evaluation requires a TranscriptML RBPNet bundle")
    if bundle.metadata is None:
        raise ValueError("RBPNet evaluation requires bundle metadata")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(calibration_bins) <= 0:
        raise ValueError("calibration_bins must be positive")
    if float(enrichment_pseudocount) <= 0:
        raise ValueError("enrichment_pseudocount must be positive")
    if int(representative_min_profile_count) < 0:
        raise ValueError("representative_min_profile_count must be non-negative")
    resolved_split, indices = resolve_rbpnet_checkpoint_indices(
        checkpoint,
        split=split,
        n_examples=int(bundle.X.shape[0]),
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()
    dataset = RBPNetDataset(
        bundle,
        max_train_jitter=0,
        training=False,
        require_full_measurement_interval=model.enrichment_enabled,
    )
    loader = _loader(dataset, indices, batch_size)
    n_evaluated = len(indices)
    profile_length = dataset.crop_length
    profile_memmaps: dict[str, np.memmap] = {}
    if save_profiles:
        for name in ("target", "control", "ip"):
            profile_memmaps[name] = np.lib.format.open_memmap(
                out / f"predicted_{name}_profiles.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_evaluated, profile_length),
            )
    else:
        for name in ("target", "control", "ip"):
            stale = out / f"predicted_{name}_profiles.npy"
            if stale.is_file():
                stale.unlink()

    log_progress(
        f"RBPNet evaluate: split={resolved_split}, examples={n_evaluated:,}, device={resolved_device}",
        enabled=progress,
    )
    rows: list[dict[str, object]] = []
    calibration_locus_rows: list[dict[str, object]] = []
    replicate_names = dataset.replicate_names
    depth_offsets = dataset.depth_offsets.astype(np.float64)
    enrichment_eta: list[float] = []
    enrichment_empirical: list[np.ndarray] = []
    enrichment_predicted: list[np.ndarray] = []
    enrichment_ip_counts: list[np.ndarray] = []
    enrichment_totals: list[np.ndarray] = []
    output_position = 0
    reporter = ProgressReporter(
        "RBPNet evaluate",
        total=len(loader),
        unit="batches",
        enabled=progress,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(resolved_device)
            output = model(
                batch.sequence,
                measurement_mask=batch.measurement_mask,
                profile_mask=batch.profile_valid_mask,
            )
            pooled_counts = batch.pooled_ip_profile.detach().cpu().numpy().astype(np.float64)
            sm_counts = batch.sminput_profile.detach().cpu().numpy().astype(np.float64)
            individual_counts = (
                batch.individual_ip_profiles.detach().cpu().numpy().astype(np.float64)
            )
            valid = batch.profile_valid_mask.detach().cpu().numpy().astype(bool)
            predicted_ip = output.ip_probs.detach().cpu().numpy().astype(np.float64)
            predicted_control = output.control_probs.detach().cpu().numpy().astype(np.float64)
            predicted_target = output.target_probs.detach().cpu().numpy().astype(np.float64)
            ip_metrics = profile_metrics(
                pooled_counts,
                predicted_ip,
                valid_mask=valid,
                control_probabilities=predicted_control,
            )
            sm_metrics = profile_metrics(
                sm_counts,
                predicted_control,
                valid_mask=valid,
            )
            ceiling = replicate_ceiling_metrics(
                individual_counts,
                valid_mask=valid,
            )
            eta_values = (
                output.enrichment_logit.detach().cpu().numpy().astype(np.float64)
                if output.enrichment_logit is not None
                else None
            )
            enrichment = None
            selection_ip = batch.ip_measurement_counts.detach().cpu().numpy().astype(np.float64)
            selection_sm = (
                batch.sminput_measurement_counts.detach().cpu().numpy().astype(np.float64)
            )
            if eta_values is not None:
                enrichment = enrichment_metrics(
                    eta_values,
                    selection_ip,
                    selection_sm,
                    depth_offsets,
                    pseudocount=enrichment_pseudocount,
                )
                enrichment_eta.extend(float(value) for value in eta_values)
                enrichment_empirical.append(enrichment["empirical_eta"])
                enrichment_predicted.append(enrichment["predicted_probability"])
                enrichment_ip_counts.append(selection_ip)
                enrichment_totals.append(enrichment["count"])

            batch_size_actual = pooled_counts.shape[0]
            if save_profiles:
                end = output_position + batch_size_actual
                profile_memmaps["target"][output_position:end] = predicted_target.astype(np.float32)
                profile_memmaps["control"][output_position:end] = predicted_control.astype(np.float32)
                profile_memmaps["ip"][output_position:end] = predicted_ip.astype(np.float32)

            for local in range(batch_size_actual):
                original_index = int(batch.indices[local].detach().cpu())
                row = _base_example_row(
                    bundle,
                    original_index,
                    output_position + local,
                    resolved_split,
                )
                row["pi"] = float(output.pi[local].detach().cpu())
                row["zero_jitter_crop_start"] = int(
                    batch.crop_start[local].detach().cpu()
                )
                row["zero_jitter_crop_end"] = (
                    row["zero_jitter_crop_start"] + profile_length
                )
                _profile_columns(row, "pooled_ip", ip_metrics, local)
                _profile_columns(row, "sminput", sm_metrics, local)
                row["selection_sminput_count"] = int(selection_sm[local])
                row["selection_ip_pooled_count"] = int(selection_ip[local].sum())
                row["sminput_effective_library_size"] = int(
                    dataset.sminput_library_size
                )

                if ceiling["jsd"].shape[1] > 0:
                    ceiling_jsd = ceiling["jsd"][local]
                    ceiling_wasserstein = ceiling["wasserstein_nt"][local]
                    ceiling_count = ceiling["count"][local]
                    jsd_mean, jsd_num, jsd_reads = _weighted_row_mean(
                        ceiling_jsd[None, :], ceiling_count[None, :]
                    )
                    wass_mean, wass_num, _ = _weighted_row_mean(
                        ceiling_wasserstein[None, :], ceiling_count[None, :]
                    )
                    row["ip_replicate_ceiling_jsd"] = _mean_finite(ceiling_jsd)
                    row["ip_replicate_ceiling_wasserstein_nt"] = _mean_finite(
                        ceiling_wasserstein
                    )
                    row["ip_replicate_ceiling_count"] = float(jsd_reads[0])
                    row["ip_replicate_ceiling_jsd_read_numerator"] = float(jsd_num[0])
                    row["ip_replicate_ceiling_wasserstein_nt_read_numerator"] = float(
                        wass_num[0]
                    )
                    for replicate, name in enumerate(replicate_names):
                        row[f"replicate_{name}_ceiling_jsd"] = float(ceiling_jsd[replicate])
                        row[f"replicate_{name}_ceiling_wasserstein_nt"] = float(
                            ceiling_wasserstein[replicate]
                        )

                for replicate, name in enumerate(replicate_names):
                    row[f"replicate_{name}_profile_count"] = int(
                        individual_counts[local, replicate].sum()
                    )
                    row[f"replicate_{name}_selection_ip_count"] = int(
                        selection_ip[local, replicate]
                    )
                    row[f"replicate_{name}_effective_library_size"] = int(
                        dataset.ip_library_sizes[replicate]
                    )
                    row[f"replicate_{name}_depth_offset"] = float(
                        depth_offsets[replicate]
                    )

                if enrichment is not None and eta_values is not None:
                    row["eta"] = float(eta_values[local])
                    valid_enrichment = enrichment["count"][local] > 0
                    nll_values = enrichment["binomial_nll"][local]
                    total_values = enrichment["count"][local]
                    info_values = enrichment[
                        "information_gain_depth_null_per_read"
                    ][local]
                    row["enrichment_binomial_nll"] = _mean_finite(nll_values)
                    row["enrichment_binomial_nll_numerator"] = float(
                        np.nansum(nll_values)
                    )
                    nll_without_constant = enrichment[
                        "binomial_nll_without_constant"
                    ][local]
                    row["enrichment_binomial_nll_without_constant"] = _mean_finite(
                        nll_without_constant
                    )
                    row[
                        "enrichment_binomial_nll_without_constant_numerator"
                    ] = float(np.nansum(nll_without_constant))
                    row["enrichment_n_observations"] = int(
                        np.count_nonzero(np.isfinite(nll_values))
                    )
                    info_mean, info_num, info_reads = _weighted_row_mean(
                        info_values[None, :], total_values[None, :]
                    )
                    row["enrichment_information_gain_depth_null_per_read"] = float(
                        info_mean[0]
                    )
                    row["enrichment_count"] = float(info_reads[0])
                    row["enrichment_information_gain_depth_null_numerator"] = float(
                        info_num[0]
                    )
                    for replicate, name in enumerate(replicate_names):
                        total = int(enrichment["count"][local, replicate])
                        row[f"replicate_{name}_enrichment_count"] = total
                        row[f"replicate_{name}_predicted_ip_fraction"] = float(
                            enrichment["predicted_probability"][local, replicate]
                        )
                        row[f"replicate_{name}_observed_ip_fraction"] = float(
                            enrichment["observed_fraction"][local, replicate]
                        )
                        row[f"replicate_{name}_empirical_eta"] = float(
                            enrichment["empirical_eta"][local, replicate]
                        )
                        row[f"replicate_{name}_binomial_nll"] = float(
                            enrichment["binomial_nll"][local, replicate]
                        )
                        row[f"replicate_{name}_binomial_nll_without_constant"] = float(
                            enrichment["binomial_nll_without_constant"][
                                local, replicate
                            ]
                        )
                        row[f"replicate_{name}_depth_null_nll"] = float(
                            enrichment["depth_null_nll"][local, replicate]
                        )
                        row[
                            f"replicate_{name}_information_gain_depth_null_per_read"
                        ] = float(info_values[replicate])
                        if valid_enrichment[replicate]:
                            calibration_locus_rows.append(
                                {
                                    "row_type": "locus",
                                    "replicate": name,
                                    "evaluation_row": output_position + local,
                                    "index": original_index,
                                    "example_id": str(bundle.ids[original_index]),
                                    "eta": float(eta_values[local]),
                                    "empirical_eta": float(
                                        enrichment["empirical_eta"][local, replicate]
                                    ),
                                    "predicted_probability": float(
                                        enrichment["predicted_probability"][local, replicate]
                                    ),
                                    "observed_fraction": float(
                                        enrichment["observed_fraction"][local, replicate]
                                    ),
                                    "ip_count": int(selection_ip[local, replicate]),
                                    "sminput_count": int(selection_sm[local]),
                                    "total_reads": total,
                                }
                            )
                rows.append(row)
            output_position += batch_size_actual
            reporter.update()
    reporter.close()
    for array in profile_memmaps.values():
        array.flush()
    if output_position != n_evaluated:
        raise RuntimeError("RBPNet evaluation did not emit every requested example")

    stratified = _stratified_metric_rows(rows, replicate_names)
    calibration = list(calibration_locus_rows)
    enrichment_correlations: dict[str, object] | None = None
    if enrichment_predicted:
        predicted = np.concatenate(enrichment_predicted, axis=0)
        empirical = np.concatenate(enrichment_empirical, axis=0)
        ip_counts = np.concatenate(enrichment_ip_counts, axis=0)
        totals = np.concatenate(enrichment_totals, axis=0)
        eta_array = np.asarray(enrichment_eta, dtype=np.float64)
        binned = calibration_rows(
            predicted,
            ip_counts,
            totals,
            replicate_names,
            n_bins=calibration_bins,
        )
        calibration.extend({"row_type": "bin", **row} for row in binned)
        enrichment_correlations = {
            "overall_locus_replicate": safe_correlations(
                np.repeat(eta_array[:, None], len(replicate_names), axis=1),
                empirical,
            ),
            "by_replicate": {
                name: safe_correlations(eta_array, empirical[:, replicate])
                for replicate, name in enumerate(replicate_names)
            },
            "empirical_eta_pseudocount": float(enrichment_pseudocount),
        }

    _write_table(rows, out / "examples.parquet")
    _write_table(stratified, out / "stratified_metrics.parquet")
    _write_table(calibration, out / "calibration.parquet", _calibration_schema())

    ip_kl = np.asarray([row["pooled_ip_kl_per_read"] for row in rows], dtype=float)
    representative = select_representative_examples(
        ip_kl,
        eligible=np.asarray(
            [
                row["pooled_ip_profile_count"] >= representative_min_profile_count
                and row["sminput_profile_count"] >= representative_min_profile_count
                for row in rows
            ],
            dtype=bool,
        ),
        seed=representative_seed,
        per_tier=representative_per_tier,
    )
    for selection in representative:
        evaluation_row = int(selection["index"])
        selection["evaluation_row"] = evaluation_row
        selection["index"] = int(rows[evaluation_row]["index"])
        selection["example_id"] = str(rows[evaluation_row]["example_id"])

    from transcriptml.plotting.rbpnet_evaluation import create_rbpnet_evaluation_plots

    plot_result = create_rbpnet_evaluation_plots(
        rows,
        calibration,
        plots_dir,
        replicate_names=replicate_names,
        representative=representative,
        dataset=dataset,
        model=model,
        device=resolved_device,
        saved_profile_paths=(
            {
                name: out / f"predicted_{name}_profiles.npy"
                for name in ("target", "control", "ip")
            }
            if save_profiles
            else None
        ),
    )

    overall_metrics = [
        row
        for row in stratified
        if row["dimension"] == "overall" and row["stratum"] == "all"
    ]
    loss_config = RBPNetLossConfig.from_config(checkpoint.get("loss_config"))
    enrichment_observations = sum(
        int(row.get("enrichment_n_observations", 0)) for row in rows
    )
    enrichment_nll_sum = sum(
        float(row.get("enrichment_binomial_nll_numerator", 0.0)) for row in rows
    )
    enrichment_nll_without_constant_sum = sum(
        float(
            row.get("enrichment_binomial_nll_without_constant_numerator", 0.0)
        )
        for row in rows
    )
    ip_nll_column = (
        "pooled_ip_multinomial_nll"
        if loss_config.include_multinomial_constant
        else "pooled_ip_multinomial_nll_without_constant"
    )
    sm_nll_column = (
        "sminput_multinomial_nll"
        if loss_config.include_multinomial_constant
        else "sminput_multinomial_nll_without_constant"
    )
    ip_nll = np.asarray(
        [row.get(ip_nll_column, np.nan) for row in rows], dtype=float
    )
    sm_nll = np.asarray(
        [row.get(sm_nll_column, np.nan) for row in rows], dtype=float
    )
    objective_components = {
        "pooled_ip_multinomial_nll": (
            float(ip_nll[np.isfinite(ip_nll)].mean())
            if np.isfinite(ip_nll).any()
            else 0.0
        ),
        "sminput_multinomial_nll": (
            float(sm_nll[np.isfinite(sm_nll)].mean())
            if np.isfinite(sm_nll).any()
            else 0.0
        ),
        "enrichment_binomial_nll": (
            (
                enrichment_nll_sum
                if loss_config.include_binomial_constant
                else enrichment_nll_without_constant_sum
            )
            / enrichment_observations
            if enrichment_observations > 0
            else 0.0
        ),
    }
    objective_loss = (
        loss_config.lambda_ip_profile
        * objective_components["pooled_ip_multinomial_nll"]
        + loss_config.lambda_sm_profile
        * objective_components["sminput_multinomial_nll"]
        + (
            loss_config.lambda_enrichment
            * objective_components["enrichment_binomial_nll"]
            if model.enrichment_enabled
            else 0.0
        )
    )
    summary = {
        "format": "transcriptml-rbpnet-evaluation",
        "format_version": "1",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "evaluated_split": resolved_split,
        "split_source": "checkpoint.splits",
        "n_examples": n_evaluated,
        "zero_jitter": True,
        "replicate_names": list(replicate_names),
        "sample_depths": {
            "sminput": {
                "name": dataset.sminput_name,
                "effective_library_size": int(dataset.sminput_library_size),
            },
            "ip": [
                {
                    "name": name,
                    "effective_library_size": int(dataset.ip_library_sizes[index]),
                    "log_library_size_ratio_vs_sminput": float(depth_offsets[index]),
                }
                for index, name in enumerate(replicate_names)
            ],
        },
        "enrichment_head_enabled": bool(model.enrichment_enabled),
        "enrichment_empirical_pseudocount": float(enrichment_pseudocount),
        "calibration_bins": int(calibration_bins),
        "profile_metric_definitions": PROFILE_METRIC_DEFINITIONS,
        "enrichment_metric_definitions": ENRICHMENT_METRIC_DEFINITIONS,
        "aggregation_definitions": {
            "locus_macro": "average replicate observations within locus, then loci equally",
            "gene_macro": "average loci within gene, then genes equally",
            "read_micro": "sum likelihood/information numerators divided by contributing reads",
        },
        "overall_metrics": overall_metrics,
        "training_objective": {
            "loss_config": loss_config.to_dict(
                enrichment_enabled=model.enrichment_enabled
            ),
            "component_columns": {
                "pooled_ip": ip_nll_column,
                "sminput": sm_nll_column,
                "enrichment": (
                    "enrichment_binomial_nll"
                    if loss_config.include_binomial_constant
                    else "enrichment_binomial_nll_without_constant"
                ),
            },
            "components": objective_components,
            "weighted_loss": objective_loss,
        },
        "enrichment_correlations": enrichment_correlations,
        "representative_examples": representative,
        "representative_sampling": {
            "seed": int(representative_seed),
            "per_performance_tertile": int(representative_per_tier),
            "minimum_pooled_ip_profile_count": int(representative_min_profile_count),
            "minimum_sminput_profile_count": int(representative_min_profile_count),
            "ranking_metric": "pooled_ip_kl_per_read",
        },
        "outputs": {
            "examples": "examples.parquet",
            "stratified_metrics": "stratified_metrics.parquet",
            "calibration": "calibration.parquet",
            "plots": "plots",
            "predicted_profiles": (
                {
                    name: f"predicted_{name}_profiles.npy"
                    for name in ("target", "control", "ip")
                }
                if save_profiles
                else None
            ),
        },
        "predicted_profile_contract": {
            "saved": bool(save_profiles),
            "dtype": "float32",
            "shape": [n_evaluated, profile_length],
            "axis_0": "evaluation_row in examples.parquet",
            "axis_1": "zero-jitter model profile position",
            "values": (
                "normalized positional probabilities over valid positions; "
                "target is latent, control predicts SMInput, and ip is the "
                "target/control mixture"
            ),
        },
        "plots": plot_result,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    log_progress(f"RBPNet evaluate: wrote report {out}", enabled=progress)
    return {
        "report_dir": str(out),
        "summary": summary,
        "indices": indices,
        "example_ids": [str(bundle.ids[index]) for index in indices],
    }
