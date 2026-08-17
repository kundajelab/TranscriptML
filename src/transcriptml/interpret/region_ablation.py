from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from transcriptml.data.region_edits import (
    base_channel_indices,
    randomize_nucleotides_inplace,
    region_bounds,
    resolve_cds_channel,
    shuffle_codons_inplace,
    shuffle_nucleotides_inplace,
    valid_length_from_bases,
)
from transcriptml.data.schemas import SequenceSchema, get_schema
from transcriptml.interpret.codon_ism import (
    CDSCodonStarts,
    find_cds_codon_starts,
    resolve_analysis_indices,
)
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_table
from transcriptml.progress import ProgressReporter, log_progress


DEFAULT_JUNCTION_COUNTS: tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
REGION_ABLATION_FAMILIES: tuple[str, ...] = (
    "5utr_shuffle",
    "5utr_random",
    "cds_nt_shuffle",
    "cds_codon_shuffle",
    "cds_random",
    "3utr_shuffle",
    "3utr_random",
    "junction_scatter",
)

_SEQUENCE_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("5utr_shuffle", "5utr", "shuffle_nucleotides"),
    ("5utr_random", "5utr", "randomize_nucleotides"),
    ("cds_nt_shuffle", "cds", "shuffle_nucleotides"),
    ("cds_codon_shuffle", "cds", "shuffle_codons"),
    ("cds_random", "cds", "randomize_nucleotides"),
    ("3utr_shuffle", "3utr", "shuffle_nucleotides"),
    ("3utr_random", "3utr", "randomize_nucleotides"),
)
_FAMILY_CODES = {family: i + 1 for i, family in enumerate(REGION_ABLATION_FAMILIES)}


@dataclass(frozen=True)
class RegionAblationConfig:
    """Configuration for repeated region and junction perturbations."""

    n_ablations: int = 100
    n_ablations_for: Mapping[str, int] = field(default_factory=dict)
    junction_counts: tuple[int, ...] = DEFAULT_JUNCTION_COUNTS
    junction_min_spacing: int = 25
    seed: int = 123

    def normalized(self) -> "RegionAblationConfig":
        """Validate values and return an immutable normalized configuration."""

        default_n = int(self.n_ablations)
        if default_n <= 0:
            raise ValueError("n_ablations must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.junction_min_spacing) <= 0:
            raise ValueError("junction_min_spacing must be positive")

        counts = tuple(int(value) for value in self.junction_counts)
        if not counts:
            raise ValueError("junction_counts must contain at least one count")
        if any(value <= 0 for value in counts):
            raise ValueError("junction_counts must contain only positive integers")
        if len(set(counts)) != len(counts):
            raise ValueError("junction_counts must be unique")

        overrides: dict[str, int] = {}
        for raw_family, raw_count in dict(self.n_ablations_for).items():
            family = str(raw_family)
            if family not in REGION_ABLATION_FAMILIES:
                raise ValueError(
                    f"Unknown region-ablation family {family!r}; expected one of "
                    f"{', '.join(REGION_ABLATION_FAMILIES)}"
                )
            count = int(raw_count)
            if count < 0:
                raise ValueError("per-family ablation counts must be non-negative")
            overrides[family] = count
        return RegionAblationConfig(
            n_ablations=default_n,
            n_ablations_for=overrides,
            junction_counts=counts,
            junction_min_spacing=int(self.junction_min_spacing),
            seed=int(self.seed),
        )

    def replicates_for(self, family: str) -> int:
        """Return the requested replicate count for one perturbation family."""

        return int(self.n_ablations_for.get(family, self.n_ablations))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration mapping."""

        return {
            "n_ablations": int(self.n_ablations),
            "n_ablations_for": {str(k): int(v) for k, v in self.n_ablations_for.items()},
            "junction_counts": [int(value) for value in self.junction_counts],
            "junction_min_spacing": int(self.junction_min_spacing),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class RegionAblationInstance:
    """One transcript-condition row aligned to region-ablation arrays."""

    instance_index: int
    seq_index: int
    sequence_id: str
    transcript_class: str
    operation: str
    region: str
    valid_length: int
    region_start: int
    region_end: int
    region_length: int
    n_replicates: int
    junction_count: int | None = None
    reference_junction_count: int | None = None
    requested_min_spacing: int | None = None
    effective_min_spacing: int | None = None


@dataclass(frozen=True)
class SkippedRegionAblation:
    """A transcript-condition omitted from the scored instance table."""

    seq_index: int
    sequence_id: str
    transcript_class: str
    operation: str
    region: str
    junction_count: int | None
    reason: str


@dataclass
class RegionAblationResult:
    """Raw and summarized repeated region-ablation predictions."""

    instances: list[RegionAblationInstance]
    skipped: list[SkippedRegionAblation]
    reference_predictions: np.ndarray
    ablation_predictions: np.ndarray
    effects: np.ndarray
    replicate_mask: np.ndarray
    mean_effects: np.ndarray
    mean_abs_effects: np.ndarray
    std_effects: np.ndarray
    analysis_indices: np.ndarray
    transcript_classes: tuple[str, ...]
    config: RegionAblationConfig
    input_shape: tuple[int, int, int]
    schema_name: str
    cds_channel_index: int
    splice_channel_index: int
    sequence_ids: tuple[str, ...]
    storage_dir: Path | None = None


def _resolve_splice_channel(schema: SequenceSchema, splice_channel: str | int | None) -> int:
    """Resolve a splice-junction channel selector to an integer index."""

    if isinstance(splice_channel, int):
        if splice_channel < 0 or splice_channel >= schema.n_channels:
            raise ValueError(
                f"splice_channel index {splice_channel} is outside schema with "
                f"{schema.n_channels} channels"
            )
        return int(splice_channel)
    if isinstance(splice_channel, str):
        try:
            return schema.channels.index(splice_channel)
        except ValueError as exc:
            raise ValueError(
                f"splice_channel {splice_channel!r} is not in schema channels {schema.channels}"
            ) from exc

    lower_to_index = {name.lower(): i for i, name in enumerate(schema.channels)}
    for name in ("splice_junction", "splice-junction", "splice", "junction"):
        if name in lower_to_index:
            return lower_to_index[name]
    for i, name in enumerate(schema.channels):
        lowered = name.lower()
        if "splice" in lowered or "junction" in lowered:
            return i
    raise ValueError("Could not infer splice-junction channel from schema; pass splice_channel explicitly")


def _metadata_reports_coding(metadata: Mapping[str, object] | None) -> bool:
    """Return whether metadata explicitly reports a positive original CDS length."""

    if metadata is None or metadata.get("cds_length") is None:
        return False
    try:
        return float(metadata["cds_length"]) > 0
    except (TypeError, ValueError):
        return False


def effective_junction_spacing(region_length: int, junction_count: int, requested_spacing: int) -> int:
    """Return the largest feasible spacing up to the requested soft target."""

    length = int(region_length)
    count = int(junction_count)
    requested = int(requested_spacing)
    if count <= 0:
        raise ValueError("junction_count must be positive")
    if requested <= 0:
        raise ValueError("requested_spacing must be positive")
    if length <= count:
        raise ValueError("region_length must be greater than junction_count")
    if count == 1:
        return requested
    maximum = max(1, (length - 2) // (count - 1))
    return min(requested, maximum)


def sample_junction_positions(
    *,
    start: int,
    end: int,
    junction_count: int,
    min_spacing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample spaced junction marks within a half-open region.

    Candidate marks are ``start`` through ``end - 2`` so every mark has at
    least one downstream nucleotide. Compressed coordinates provide a
    rejection-free bijection to layouts obeying the requested separation.
    """

    region_start = int(start)
    region_end = int(end)
    count = int(junction_count)
    spacing = int(min_spacing)
    region_length = region_end - region_start
    if region_length <= count:
        raise ValueError("region length must be greater than junction_count")
    if count <= 0:
        raise ValueError("junction_count must be positive")
    if spacing <= 0:
        raise ValueError("min_spacing must be positive")

    candidate_count = region_length - 1
    if count == 1:
        return np.asarray(
            [region_start + int(rng.integers(0, candidate_count))],
            dtype=np.int64,
        )
    maximum_spacing = max(1, (region_length - 2) // (count - 1))
    if spacing > maximum_spacing:
        raise ValueError(
            f"min_spacing={spacing} is infeasible for {count} junctions in length {region_length}"
        )
    compressed_count = candidate_count - (spacing - 1) * (count - 1)
    compressed = np.sort(rng.choice(compressed_count, size=count, replace=False))
    expanded = compressed + np.arange(count, dtype=np.int64) * (spacing - 1)
    return expanded.astype(np.int64, copy=False) + region_start


def scatter_junctions_inplace(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    splice_channel: int,
    junction_count: int,
    min_spacing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Clear and replace junction marks in one region, returning new positions."""

    channel = int(splice_channel)
    x[channel, int(start) : int(end)] = 0
    positions = sample_junction_positions(
        start=start,
        end=end,
        junction_count=junction_count,
        min_spacing=min_spacing,
        rng=rng,
    )
    x[channel, positions] = 1
    return positions


def _sequence_ids_digest(sequence_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sequence_id in sequence_ids:
        digest.update(str(sequence_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _allocate_array(
    storage_dir: Path | None,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
    fill_value: float | bool,
) -> np.ndarray:
    if storage_dir is not None and all(dimension > 0 for dimension in shape):
        storage_dir.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            storage_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
    else:
        out = np.empty(shape, dtype=dtype)
    out[...] = fill_value
    return out


def _predict(predictor: Predictor, X: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
    try:
        values = predictor.predict(X, batch_size=batch_size)
    except TypeError:
        values = predictor.predict(X)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _reference_batch(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
    first = int(indices[0])
    last = int(indices[-1])
    if np.array_equal(indices, np.arange(first, last + 1, dtype=np.int64)):
        return arr[first : last + 1]
    return arr[indices]


def _replicate_rng(
    seed: int,
    seq_index: int,
    operation: str,
    junction_count: int | None,
    replicate_index: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(seq_index),
                int(_FAMILY_CODES[operation]),
                0 if junction_count is None else int(junction_count),
                int(replicate_index),
            ]
        )
    )


def region_ablation(
    X: np.ndarray,
    predictor: Predictor,
    *,
    schema: str | SequenceSchema = "saluki6",
    sequence_ids: Sequence[str] | None = None,
    metadata: Sequence[Mapping[str, object]] | None = None,
    config: RegionAblationConfig | None = None,
    valid_lengths: Sequence[int] | None = None,
    cds_channel: str | int | None = None,
    splice_channel: str | int | None = None,
    reference_batch_size: int | None = None,
    mutation_batch_size: int = 512,
    sequence_indices: Sequence[int] | None = None,
    sequence_start: int | None = None,
    sequence_end: int | None = None,
    sequence_shard_index: int | None = None,
    sequence_shards: int | None = None,
    storage_dir: str | Path | None = None,
    progress: bool = True,
) -> RegionAblationResult:
    """Run repeated region sequence edits and exon-junction density scans."""

    arr = np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    resolved_schema = get_schema(schema)
    if arr.shape[1] < resolved_schema.n_channels:
        raise ValueError(
            f"X has {arr.shape[1]} channels, but schema {resolved_schema.name!r} "
            f"expects {resolved_schema.n_channels}"
        )
    cfg = (config or RegionAblationConfig()).normalized()
    if int(mutation_batch_size) <= 0:
        raise ValueError("mutation_batch_size must be positive")

    n_sequences = int(arr.shape[0])
    ids = tuple(str(i) for i in range(n_sequences)) if sequence_ids is None else tuple(map(str, sequence_ids))
    if len(ids) != n_sequences:
        raise ValueError("sequence_ids length must match X.shape[0]")
    if metadata is not None and len(metadata) != n_sequences:
        raise ValueError("metadata length must match X.shape[0]")

    base_channels = base_channel_indices(resolved_schema)
    cds_channel_index = resolve_cds_channel(resolved_schema, cds_channel)
    splice_channel_index = _resolve_splice_channel(resolved_schema, splice_channel)
    if splice_channel_index in set(base_channels.tolist()):
        raise ValueError("splice_channel must not select a nucleotide base channel")
    if splice_channel_index == cds_channel_index:
        raise ValueError("splice_channel and cds_channel must select different channels")

    analysis_indices = resolve_analysis_indices(
        n_sequences,
        sequence_indices=sequence_indices,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        sequence_shard_index=sequence_shard_index,
        sequence_shards=sequence_shards,
    )
    full_lengths = None if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    if full_lengths is not None:
        if full_lengths.shape != (n_sequences,):
            raise ValueError(f"valid_lengths must have shape ({n_sequences},)")
        if np.any(full_lengths < 0) or np.any(full_lengths > arr.shape[-1]):
            raise ValueError("valid_lengths entries are outside the encoded sequence length")

    instances: list[RegionAblationInstance] = []
    skipped: list[SkippedRegionAblation] = []
    transcript_classes: list[str] = []
    cds_by_sequence: dict[int, CDSCodonStarts] = {}
    valid_length_by_sequence: dict[int, int] = {}

    def add_skip(
        seq_index: int,
        transcript_class: str,
        operation: str,
        region: str,
        reason: str,
        junction_count: int | None = None,
    ) -> None:
        skipped.append(
            SkippedRegionAblation(
                seq_index=seq_index,
                sequence_id=ids[seq_index],
                transcript_class=transcript_class,
                operation=operation,
                region=region,
                junction_count=junction_count,
                reason=reason,
            )
        )

    reporter = ProgressReporter(
        "region-ablation: enumerate conditions",
        total=int(analysis_indices.size),
        unit="transcripts",
        enabled=progress,
    )
    for seq_index_raw in analysis_indices:
        seq_index = int(seq_index_raw)
        x = np.asarray(arr[seq_index])
        valid_length = (
            int(full_lengths[seq_index])
            if full_lengths is not None
            else valid_length_from_bases(x, base_channels)
        )
        valid_length = min(max(valid_length, 0), int(arr.shape[-1]))
        valid_length_by_sequence[seq_index] = valid_length
        cds = find_cds_codon_starts(
            x,
            resolved_schema,
            valid_length=valid_length,
            cds_channel=cds_channel_index,
        )
        has_cds = cds.starts.size > 0 and cds.cds_length >= 3
        row_metadata = None if metadata is None else metadata[seq_index]
        if has_cds:
            transcript_class = "coding"
            cds_by_sequence[seq_index] = cds
        elif _metadata_reports_coding(row_metadata):
            transcript_class = "coding_unresolved"
        else:
            transcript_class = "noncoding"
        transcript_classes.append(transcript_class)

        if transcript_class == "coding_unresolved":
            for family, region, _ in _SEQUENCE_FAMILIES:
                if cfg.replicates_for(family) > 0:
                    add_skip(seq_index, transcript_class, family, region, "represented_cds_missing")
            if cfg.replicates_for("junction_scatter") > 0:
                for count in cfg.junction_counts:
                    add_skip(
                        seq_index,
                        transcript_class,
                        "junction_scatter",
                        "cds",
                        "represented_cds_missing",
                        count,
                    )
            reporter.update()
            continue

        if transcript_class == "coding":
            for family, region, _ in _SEQUENCE_FAMILIES:
                n_replicates = cfg.replicates_for(family)
                if n_replicates == 0:
                    continue
                bounds = region_bounds(
                    region, valid_length=valid_length, cds=cds
                )
                if bounds is None or bounds[1] <= bounds[0]:
                    add_skip(seq_index, transcript_class, family, region, "empty_region")
                    continue
                start, end = bounds
                instances.append(
                    RegionAblationInstance(
                        instance_index=len(instances),
                        seq_index=seq_index,
                        sequence_id=ids[seq_index],
                        transcript_class=transcript_class,
                        operation=family,
                        region=region,
                        valid_length=valid_length,
                        region_start=int(start),
                        region_end=int(end),
                        region_length=int(end - start),
                        n_replicates=n_replicates,
                    )
                )
            junction_region = "cds"
            junction_bounds = region_bounds(
                "cds", valid_length=valid_length, cds=cds
            )
        else:
            junction_region = "transcript"
            junction_bounds = (0, valid_length)

        junction_replicates = cfg.replicates_for("junction_scatter")
        if junction_replicates > 0:
            assert junction_bounds is not None
            start, end = junction_bounds
            region_length = int(end - start)
            reference_junction_count = int(
                np.count_nonzero(x[splice_channel_index, int(start) : int(end)] > 0)
            )
            for count in cfg.junction_counts:
                if region_length <= count:
                    add_skip(
                        seq_index,
                        transcript_class,
                        "junction_scatter",
                        junction_region,
                        "region_length_not_greater_than_junction_count",
                        count,
                    )
                    continue
                spacing = effective_junction_spacing(
                    region_length,
                    count,
                    cfg.junction_min_spacing,
                )
                instances.append(
                    RegionAblationInstance(
                        instance_index=len(instances),
                        seq_index=seq_index,
                        sequence_id=ids[seq_index],
                        transcript_class=transcript_class,
                        operation="junction_scatter",
                        region=junction_region,
                        valid_length=valid_length,
                        region_start=int(start),
                        region_end=int(end),
                        region_length=region_length,
                        n_replicates=junction_replicates,
                        junction_count=int(count),
                        reference_junction_count=reference_junction_count,
                        requested_min_spacing=cfg.junction_min_spacing,
                        effective_min_spacing=spacing,
                    )
                )
        reporter.update()
    reporter.close(extra=f"{len(instances)} conditions")

    storage_path = None if storage_dir is None else Path(storage_dir)
    n_instances = len(instances)
    max_replicates = max((instance.n_replicates for instance in instances), default=0)
    reference_out = _allocate_array(
        storage_path, "reference_predictions", (n_instances,), np.float32, np.nan
    )
    ablation_out = _allocate_array(
        storage_path,
        "ablation_predictions",
        (n_instances, max_replicates),
        np.float32,
        np.nan,
    )
    effects_out = _allocate_array(
        storage_path, "effects", (n_instances, max_replicates), np.float32, np.nan
    )
    mask_out = _allocate_array(
        storage_path, "replicate_mask", (n_instances, max_replicates), np.bool_, False
    )
    for instance in instances:
        mask_out[instance.instance_index, : instance.n_replicates] = True

    X_reference = _reference_batch(arr, analysis_indices)
    log_progress(
        f"region-ablation: predicting {analysis_indices.size} reference sequences",
        enabled=progress,
    )
    selected_reference = _predict(predictor, X_reference, batch_size=reference_batch_size)
    if selected_reference.shape != (analysis_indices.size,):
        raise ValueError("predictor must return one scalar prediction per reference sequence")
    reference_by_sequence = {
        int(seq_index): float(prediction)
        for seq_index, prediction in zip(analysis_indices.tolist(), selected_reference.tolist())
    }
    for instance in instances:
        reference_out[instance.instance_index] = reference_by_sequence[instance.seq_index]

    mutant_batch: list[np.ndarray] = []
    pending: list[tuple[int, int]] = []

    def flush_mutants() -> None:
        if not mutant_batch:
            return
        predictions = _predict(
            predictor,
            np.stack(mutant_batch, axis=0),
            batch_size=mutation_batch_size,
        )
        if predictions.shape != (len(pending),):
            raise ValueError("predictor must return one scalar prediction per mutant sequence")
        for prediction, (instance_index, replicate_index) in zip(predictions, pending):
            value = np.float32(prediction)
            ablation_out[instance_index, replicate_index] = value
            effects_out[instance_index, replicate_index] = np.float32(
                value - reference_out[instance_index]
            )
        mutant_batch.clear()
        pending.clear()

    total_mutants = sum(instance.n_replicates for instance in instances)
    mutation_reporter = ProgressReporter(
        "region-ablation: predict mutants",
        total=total_mutants,
        unit="mutants",
        enabled=progress,
    )
    for instance in instances:
        for replicate_index in range(instance.n_replicates):
            mutant = np.array(arr[instance.seq_index], copy=True)
            rng = _replicate_rng(
                cfg.seed,
                instance.seq_index,
                instance.operation,
                instance.junction_count,
                replicate_index,
            )
            if instance.operation == "junction_scatter":
                assert instance.junction_count is not None
                assert instance.effective_min_spacing is not None
                scatter_junctions_inplace(
                    mutant,
                    start=instance.region_start,
                    end=instance.region_end,
                    splice_channel=splice_channel_index,
                    junction_count=instance.junction_count,
                    min_spacing=instance.effective_min_spacing,
                    rng=rng,
                )
            elif instance.operation == "cds_codon_shuffle":
                shuffle_codons_inplace(
                    mutant,
                    cds=cds_by_sequence[instance.seq_index],
                    base_channels=base_channels,
                    rng=rng,
                )
            elif instance.operation.endswith("_shuffle"):
                shuffle_nucleotides_inplace(
                    mutant,
                    start=instance.region_start,
                    end=instance.region_end,
                    base_channels=base_channels,
                    rng=rng,
                )
            elif instance.operation.endswith("_random"):
                randomize_nucleotides_inplace(
                    mutant,
                    start=instance.region_start,
                    end=instance.region_end,
                    base_channels=base_channels,
                    rng=rng,
                )
            else:  # pragma: no cover - guarded by the fixed family table.
                raise RuntimeError(f"Unsupported region-ablation operation {instance.operation!r}")
            mutant_batch.append(mutant)
            pending.append((instance.instance_index, replicate_index))
            mutation_reporter.update()
            if len(mutant_batch) >= int(mutation_batch_size):
                flush_mutants()
    flush_mutants()
    mutation_reporter.close(extra=f"{total_mutants} mutants")

    mean_effects = _allocate_array(
        storage_path, "mean_effects", (n_instances,), np.float32, np.nan
    )
    mean_abs_effects = _allocate_array(
        storage_path, "mean_abs_effects", (n_instances,), np.float32, np.nan
    )
    std_effects = _allocate_array(
        storage_path, "std_effects", (n_instances,), np.float32, np.nan
    )
    for instance in instances:
        values = np.asarray(
            effects_out[instance.instance_index, : instance.n_replicates],
            dtype=np.float64,
        )
        mean_effects[instance.instance_index] = np.float32(values.mean())
        mean_abs_effects[instance.instance_index] = np.float32(np.abs(values).mean())
        std_effects[instance.instance_index] = np.float32(values.std(ddof=0))

    for values in (
        reference_out,
        ablation_out,
        effects_out,
        mask_out,
        mean_effects,
        mean_abs_effects,
        std_effects,
    ):
        if hasattr(values, "flush"):
            values.flush()

    return RegionAblationResult(
        instances=instances,
        skipped=skipped,
        reference_predictions=reference_out,
        ablation_predictions=ablation_out,
        effects=effects_out,
        replicate_mask=mask_out,
        mean_effects=mean_effects,
        mean_abs_effects=mean_abs_effects,
        std_effects=std_effects,
        analysis_indices=analysis_indices.astype(np.int64, copy=False),
        transcript_classes=tuple(transcript_classes),
        config=cfg,
        input_shape=tuple(int(value) for value in arr.shape),
        schema_name=resolved_schema.name,
        cds_channel_index=cds_channel_index,
        splice_channel_index=splice_channel_index,
        sequence_ids=tuple(ids[int(index)] for index in analysis_indices),
        storage_dir=storage_path,
    )


def _persist_array(path: Path, values: np.ndarray) -> None:
    if isinstance(values, np.memmap) and getattr(values, "filename", None) is not None:
        source = Path(str(values.filename))
        if source.resolve() == path.resolve():
            values.flush()
            return
    np.save(path, np.asarray(values))


def save_region_ablation_result(
    result: RegionAblationResult,
    out_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
    progress: bool = True,
) -> None:
    """Save raw arrays, condition tables, and reproducibility metadata."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"region-ablation: saving results to {out}", enabled=progress)
    arrays = {
        "reference_predictions": result.reference_predictions,
        "ablation_predictions": result.ablation_predictions,
        "effects": result.effects,
        "replicate_mask": result.replicate_mask,
        "mean_effects": result.mean_effects,
        "mean_abs_effects": result.mean_abs_effects,
        "std_effects": result.std_effects,
    }
    for name, values in arrays.items():
        _persist_array(out / f"{name}.npy", values)
    save_table(out / "instances.csv", result.instances)
    save_table(out / "skipped.csv", result.skipped)

    operation_counts = Counter(instance.operation for instance in result.instances)
    skip_counts = Counter(row.reason for row in result.skipped)
    class_counts = Counter(result.transcript_classes)
    summary = {
        "analysis": "region_ablation",
        "effect_definition": "ablation_prediction - reference_prediction",
        "raw_replicates_saved": True,
        "inactive_replicate_fill_value": "NaN",
        "coordinate_convention": "zero_based_half_open",
        "junction_mark_definition": "nucleotide immediately upstream of exon-exon boundary",
        "junction_candidate_interval": "[region_start, region_end - 1)",
        "junction_spacing_policy": "soft target relaxed to maximum feasible separation",
        "junction_sampling": "uniform compressed-coordinate layouts without replacement",
        "seed_policy": "SeedSequence(seed, seq_index, family_code, junction_count_or_0, replicate_index)",
        "config": result.config.to_dict(),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset": str(dataset) if dataset is not None else None,
        "input_shape": list(result.input_shape),
        "schema": result.schema_name,
        "cds_channel_index": int(result.cds_channel_index),
        "splice_channel_index": int(result.splice_channel_index),
        "analysis_sequence_indices": [int(value) for value in result.analysis_indices],
        "sequence_ids_sha256": _sequence_ids_digest(result.sequence_ids),
        "n_sequences": int(result.analysis_indices.size),
        "n_instances": len(result.instances),
        "n_mutants": int(result.replicate_mask.sum()),
        "n_skipped": len(result.skipped),
        "operation_counts": dict(sorted(operation_counts.items())),
        "transcript_class_counts": dict(sorted(class_counts.items())),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "arrays": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in arrays.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("region-ablation: done", enabled=progress)
