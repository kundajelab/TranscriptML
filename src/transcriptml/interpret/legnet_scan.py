from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from transcriptml.data.region_edits import resolve_cds_channel
from transcriptml.data.schemas import SequenceSchema, get_schema
from transcriptml.interpret.codon_ism import find_cds_codon_starts
from transcriptml.interpret.predictor import Predictor
from transcriptml.progress import ProgressReporter, log_progress


_ANNOTATED_REGIONS = ("5utr", "cds", "3utr")


@dataclass(frozen=True)
class LegNetWindowInstance:
    """One endogenous transcript window presented to LegNet."""

    instance_index: int
    transcript_index: int
    transcript_id: str
    region: str
    transcript_start: int
    transcript_end: int
    encoded_start: int
    encoded_end: int
    unpadded_length: int
    padding_length: int
    coordinate_offset: int
    coordinate_source: str


@dataclass(frozen=True)
class RegionScanOutcome:
    """Enumeration outcome for one transcript and requested region."""

    transcript_index: int
    region: str
    instance_start: int
    instance_end: int
    status: str
    skip_reason: str | None = None


@dataclass
class LegNetScanResult:
    """Predictions and provenance for an endogenous LegNet window scan."""

    scores: np.ndarray
    instances: list[LegNetWindowInstance]
    region_outcomes: list[RegionScanOutcome]
    sequence_ids: list[str]
    metadata: list[Mapping[str, Any]]
    sequences: np.ndarray | None
    regions: tuple[str, ...]
    window_size: int
    stride: int
    input_shape: tuple[int, int, int]


def normalize_scan_regions(regions: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize region aliases and expand ``all`` deterministically."""

    if regions is None:
        tokens = ["3utr"]
    elif isinstance(regions, str):
        tokens = [token for token in regions.replace(";", ",").split(",") if token.strip()]
    else:
        tokens = [str(token) for token in regions]
    if not tokens:
        raise ValueError("regions must contain at least one region")

    aliases = {
        "5utr": "5utr",
        "utr5": "5utr",
        "fiveutr": "5utr",
        "fiveprimeutr": "5utr",
        "cds": "cds",
        "coding": "cds",
        "codingsequence": "cds",
        "3utr": "3utr",
        "utr3": "3utr",
        "threeutr": "3utr",
        "threeprimeutr": "3utr",
        "transcript": "transcript",
        "full": "transcript",
        "fulltranscript": "transcript",
        "wholetranscript": "transcript",
        "all": "all",
        "annotated": "all",
    }
    normalized: list[str] = []
    for token in tokens:
        key = (
            str(token)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace("'", "")
            .replace('"', "")
            .replace(" ", "")
        )
        try:
            region = aliases[key]
        except KeyError as exc:
            raise ValueError("regions must be drawn from: 5utr, cds, 3utr, all, transcript") from exc
        expanded = _ANNOTATED_REGIONS if region == "all" else (region,)
        for item in expanded:
            if item not in normalized:
                normalized.append(item)
    if "transcript" in normalized and len(normalized) > 1:
        raise ValueError("transcript cannot be combined with annotated regions")
    return tuple(normalized)


def generate_scan_windows(
    region_start: int,
    region_end: int,
    window_size: int,
    stride: int,
) -> tuple[tuple[int, int], ...]:
    """Generate full windows followed by at most one right-padded tail window."""

    start = int(region_start)
    end = int(region_end)
    width = int(window_size)
    step = int(stride)
    if start < 0 or end < start:
        raise ValueError("region bounds must satisfy 0 <= start <= end")
    if width <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0 or step > width:
        raise ValueError("stride must satisfy 1 <= stride <= window_size")
    length = end - start
    if length <= 0:
        return ()
    if length <= width:
        return ((start, end),)

    full_starts = list(range(start, end - width + 1, step))
    windows = [(window_start, window_start + width) for window_start in full_starts]
    last_start = full_starts[-1]
    if last_start + width < end:
        tail_start = last_start + step
        windows.append((tail_start, end))
    return tuple(windows)


def _canonical_base_channels(schema: SequenceSchema) -> np.ndarray:
    by_letter: dict[str, int] = {}
    for name in schema.base_channels:
        if name not in schema.channels:
            raise ValueError(f"Base channel {name!r} is not present in schema channels {schema.channels}")
        letter = name.upper().replace("T", "U")
        if letter not in {"A", "C", "G", "U"} or letter in by_letter:
            raise ValueError("scan-legnet requires exactly one A, C, G, and U/T base channel")
        by_letter[letter] = schema.channels.index(name)
    if set(by_letter) != {"A", "C", "G", "U"}:
        raise ValueError("scan-legnet requires exactly one A, C, G, and U/T base channel")
    return np.asarray([by_letter[base] for base in "ACGU"], dtype=np.int64)


def _optional_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {key!r} must be an integer or null, got {value!r}") from exc


def _represented_coordinates(
    x: np.ndarray,
    metadata: Mapping[str, Any],
    base_channels: np.ndarray,
) -> tuple[int, int, str]:
    encoded_width = int(x.shape[-1])
    transcript_length = _optional_int(metadata, "transcript_length")
    if transcript_length is not None:
        if transcript_length < 0:
            raise ValueError("metadata transcript_length must be non-negative")
        valid_length = min(transcript_length, encoded_width)
        return valid_length, max(0, transcript_length - encoded_width), "transcript_metadata"

    base = np.asarray(x[base_channels])
    positions = np.flatnonzero(np.any(base != 0, axis=0))
    valid_length = int(positions[-1] + 1) if positions.size else 0
    return valid_length, 0, "encoded_fallback"


def _metadata_cds_bounds(
    metadata: Mapping[str, Any],
    *,
    transcript_length: int | None,
) -> tuple[int, int] | None:
    cds_start = _optional_int(metadata, "cds_start")
    cds_end = _optional_int(metadata, "cds_end")
    if cds_start is None and cds_end is None:
        return None
    if cds_start is None or cds_end is None:
        raise ValueError("metadata must provide both cds_start and cds_end")
    upper = transcript_length if transcript_length is not None else cds_end
    if cds_start < 0 or cds_end <= cds_start or cds_end > upper:
        raise ValueError(
            "metadata CDS bounds must be zero-based half-open coordinates within transcript_length"
        )
    return cds_start, cds_end


def _region_bounds(
    x: np.ndarray,
    *,
    region: str,
    valid_length: int,
    coordinate_offset: int,
    metadata: Mapping[str, Any],
    schema: SequenceSchema,
    cds_channel: str | int | None,
) -> tuple[int, int] | None:
    if region == "transcript":
        return 0, valid_length

    transcript_length = _optional_int(metadata, "transcript_length")
    exact_cds = _metadata_cds_bounds(metadata, transcript_length=transcript_length)
    if exact_cds is not None:
        cds_start, cds_end = exact_cds
        original_end = coordinate_offset + valid_length
        original_bounds = {
            "5utr": (0, cds_start),
            "cds": (cds_start, cds_end),
            "3utr": (cds_end, transcript_length if transcript_length is not None else original_end),
        }[region]
        clipped_start = max(original_bounds[0], coordinate_offset)
        clipped_end = min(original_bounds[1], original_end)
        if clipped_end <= clipped_start:
            return clipped_start - coordinate_offset, clipped_start - coordinate_offset
        return clipped_start - coordinate_offset, clipped_end - coordinate_offset

    if x.shape[0] < 5:
        return None
    try:
        cds = find_cds_codon_starts(
            x,
            schema,
            valid_length=valid_length,
            cds_channel=cds_channel,
        )
    except ValueError:
        return None
    if cds.starts.size == 0 or cds.cds_length < 3:
        return None
    start = max(0, int(cds.cds_start))
    end = min(valid_length, int(cds.cds_end) + 1)
    if region == "5utr":
        return 0, start
    if region == "cds":
        return start, end
    return end, valid_length


def _array_at_path(array: np.ndarray, path: Path) -> bool:
    filename = getattr(array, "filename", None)
    return filename is not None and Path(str(filename)).resolve() == path.resolve()


def _allocate_array(path: Path | None, *, shape: tuple[int, ...], dtype: np.dtype | type) -> np.ndarray:
    if path is None:
        return np.empty(shape, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(dimension == 0 for dimension in shape):
        array = np.empty(shape, dtype=dtype)
        np.save(path, array)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def scan_legnet_windows(
    X: np.ndarray,
    predictor: Predictor,
    *,
    window_size: int,
    stride: int | None = None,
    regions: str | Sequence[str] | None = None,
    schema: str | SequenceSchema = "saluki6",
    sequence_ids: Sequence[str] | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    cds_channel: str | int | None = None,
    batch_size: int | None = None,
    save_sequences: bool = False,
    storage_dir: str | Path | None = None,
    progress: bool = True,
) -> LegNetScanResult:
    """Extract endogenous transcript windows and predict raw LegNet scores."""

    arr = np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    width = int(window_size)
    step = max(1, width // 4) if stride is None else int(stride)
    if width <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0 or step > width:
        raise ValueError("stride must satisfy 1 <= stride <= window_size")
    normalized_regions = normalize_scan_regions(regions)
    resolved_schema = get_schema(schema)
    base_channels = _canonical_base_channels(resolved_schema)
    if arr.shape[1] != resolved_schema.n_channels:
        raise ValueError(
            f"X has {arr.shape[1]} channels but schema {resolved_schema.name!r} "
            f"declares {resolved_schema.n_channels}"
        )
    if any(region != "transcript" for region in normalized_regions) and cds_channel is not None:
        resolve_cds_channel(resolved_schema, cds_channel)

    n_transcripts = int(arr.shape[0])
    ids = [str(index) for index in range(n_transcripts)] if sequence_ids is None else [str(x) for x in sequence_ids]
    if len(ids) != n_transcripts:
        raise ValueError("sequence_ids length must match X.shape[0]")
    rows = [dict() for _ in range(n_transcripts)] if metadata is None else [dict(row) for row in metadata]
    if len(rows) != n_transcripts:
        raise ValueError("metadata length must match X.shape[0]")

    instances: list[LegNetWindowInstance] = []
    outcomes: list[RegionScanOutcome] = []
    for transcript_index in range(n_transcripts):
        x = arr[transcript_index]
        row_metadata = rows[transcript_index]
        valid_length, coordinate_offset, coordinate_source = _represented_coordinates(
            x, row_metadata, base_channels
        )
        for region in normalized_regions:
            instance_start = len(instances)
            bounds = _region_bounds(
                x,
                region=region,
                valid_length=valid_length,
                coordinate_offset=coordinate_offset,
                metadata=row_metadata,
                schema=resolved_schema,
                cds_channel=cds_channel,
            )
            if bounds is None:
                outcomes.append(
                    RegionScanOutcome(
                        transcript_index,
                        region,
                        instance_start,
                        instance_start,
                        "skipped",
                        "unresolved_region_annotation",
                    )
                )
                continue
            windows = generate_scan_windows(bounds[0], bounds[1], width, step)
            if not windows:
                outcomes.append(
                    RegionScanOutcome(
                        transcript_index,
                        region,
                        instance_start,
                        instance_start,
                        "skipped",
                        "empty_region",
                    )
                )
                continue
            for encoded_start, encoded_end in windows:
                unpadded_length = encoded_end - encoded_start
                instances.append(
                    LegNetWindowInstance(
                        instance_index=len(instances),
                        transcript_index=transcript_index,
                        transcript_id=ids[transcript_index],
                        region=region,
                        transcript_start=coordinate_offset + encoded_start,
                        transcript_end=coordinate_offset + encoded_end,
                        encoded_start=encoded_start,
                        encoded_end=encoded_end,
                        unpadded_length=unpadded_length,
                        padding_length=width - unpadded_length,
                        coordinate_offset=coordinate_offset,
                        coordinate_source=coordinate_source,
                    )
                )
            outcomes.append(
                RegionScanOutcome(
                    transcript_index,
                    region,
                    instance_start,
                    len(instances),
                    "scored",
                )
            )

    storage = Path(storage_dir) if storage_dir is not None else None
    scores_path = None if storage is None else storage / "scores.npy"
    sequences_path = None if storage is None or not save_sequences else storage / "sequences.npy"
    scores = _allocate_array(scores_path, shape=(len(instances),), dtype=np.float32)
    sequences = (
        _allocate_array(sequences_path, shape=(len(instances), 4, width), dtype=np.uint8)
        if save_sequences
        else None
    )

    reporter = ProgressReporter(
        "scan-legnet: predict windows",
        total=len(instances),
        unit="windows",
        enabled=progress,
    )
    prediction_batch_size = int(batch_size or predictor.batch_size)
    if prediction_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for batch_start in range(0, len(instances), prediction_batch_size):
        batch_instances = instances[batch_start : batch_start + prediction_batch_size]
        batch = np.zeros((len(batch_instances), 4, width), dtype=np.uint8)
        for offset, instance in enumerate(batch_instances):
            source = arr[
                instance.transcript_index,
                base_channels,
                instance.encoded_start : instance.encoded_end,
            ]
            batch[offset, :, : instance.unpadded_length] = source
        predictions = predictor.predict(batch, batch_size=prediction_batch_size)
        if predictions.shape != (len(batch_instances),):
            raise ValueError(
                "LegNet predictor must return one scalar per window; "
                f"got shape {predictions.shape}"
            )
        batch_end = batch_start + len(batch_instances)
        scores[batch_start:batch_end] = predictions.astype(np.float32, copy=False)
        if sequences is not None:
            sequences[batch_start:batch_end] = batch
        reporter.update(len(batch_instances))
    reporter.close()
    if isinstance(scores, np.memmap):
        scores.flush()
    if isinstance(sequences, np.memmap):
        sequences.flush()

    return LegNetScanResult(
        scores=scores,
        instances=instances,
        region_outcomes=outcomes,
        sequence_ids=ids,
        metadata=rows,
        sequences=sequences,
        regions=normalized_regions,
        window_size=width,
        stride=step,
        input_shape=tuple(int(value) for value in arr.shape),
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Mapping, list, tuple, set, np.ndarray)):
        if isinstance(value, set):
            value = sorted(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _metadata_columns(metadata: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"metadata.{key}" for key in sorted({str(key) for row in metadata for key in row})]


def _add_metadata(row: dict[str, Any], metadata: Mapping[str, Any], columns: Sequence[str]) -> None:
    for column in columns:
        row[column] = _csv_value(metadata.get(column[len("metadata.") :]))


def _statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n_windows": 0,
            "predicted_stability_mean": "",
            "predicted_stability_median": "",
            "predicted_stability_max": "",
            "predicted_stability_min": "",
            "predicted_stability_std": "",
        }
    return {
        "n_windows": int(array.size),
        "predicted_stability_mean": float(np.mean(array)),
        "predicted_stability_median": float(np.median(array)),
        "predicted_stability_max": float(np.max(array)),
        "predicted_stability_min": float(np.min(array)),
        "predicted_stability_std": float(np.std(array, ddof=0)),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_legnet_scan_result(
    result: LegNetScanResult,
    out_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
    progress: bool = True,
) -> None:
    """Write aligned arrays, instance tables, aggregate tables, and provenance."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"scan-legnet: saving results to {out}", enabled=progress)
    scores_path = out / "scores.npy"
    if not _array_at_path(result.scores, scores_path):
        np.save(scores_path, np.asarray(result.scores, dtype=np.float32))
    if result.sequences is not None:
        sequences_path = out / "sequences.npy"
        if not _array_at_path(result.sequences, sequences_path):
            np.save(sequences_path, np.asarray(result.sequences, dtype=np.uint8))

    metadata_columns = _metadata_columns(result.metadata)
    instance_fields = [
        "instance_index",
        "transcript_index",
        "transcript_id",
        "region",
        "transcript_start",
        "transcript_end",
        "encoded_start",
        "encoded_end",
        "unpadded_length",
        "padding_length",
        "coordinate_offset",
        "coordinate_source",
        "predicted_stability",
        *metadata_columns,
    ]
    with (out / "instances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=instance_fields)
        writer.writeheader()
        for instance in result.instances:
            row = {
                "instance_index": instance.instance_index,
                "transcript_index": instance.transcript_index,
                "transcript_id": instance.transcript_id,
                "region": instance.region,
                "transcript_start": instance.transcript_start,
                "transcript_end": instance.transcript_end,
                "encoded_start": instance.encoded_start,
                "encoded_end": instance.encoded_end,
                "unpadded_length": instance.unpadded_length,
                "padding_length": instance.padding_length,
                "coordinate_offset": instance.coordinate_offset,
                "coordinate_source": instance.coordinate_source,
                "predicted_stability": float(result.scores[instance.instance_index]),
            }
            _add_metadata(row, result.metadata[instance.transcript_index], metadata_columns)
            writer.writerow(row)

    statistic_fields = [
        "n_windows",
        "predicted_stability_mean",
        "predicted_stability_median",
        "predicted_stability_max",
        "predicted_stability_min",
        "predicted_stability_std",
    ]
    outcome_by_transcript: dict[int, list[RegionScanOutcome]] = {
        index: [] for index in range(len(result.sequence_ids))
    }
    for outcome in result.region_outcomes:
        outcome_by_transcript[outcome.transcript_index].append(outcome)

    transcript_rows: list[dict[str, Any]] = []
    for transcript_index, transcript_id in enumerate(result.sequence_ids):
        outcomes = outcome_by_transcript[transcript_index]
        scored = [outcome for outcome in outcomes if outcome.status == "scored"]
        first = min((outcome.instance_start for outcome in scored), default=0)
        last = max((outcome.instance_end for outcome in scored), default=0)
        reasons = [f"{outcome.region}:{outcome.skip_reason}" for outcome in outcomes if outcome.skip_reason]
        if not scored:
            status = "skipped"
        elif reasons:
            status = "partial"
        else:
            status = "scored"
        row = {
            "transcript_index": transcript_index,
            "transcript_id": transcript_id,
            "regions": ",".join(result.regions),
            "status": status,
            "skip_reason": ";".join(reasons),
            **_statistics(result.scores[first:last] if scored else np.empty(0)),
        }
        _add_metadata(row, result.metadata[transcript_index], metadata_columns)
        transcript_rows.append(row)
    transcript_fields = [
        "transcript_index",
        "transcript_id",
        "regions",
        "status",
        "skip_reason",
        *statistic_fields,
        *metadata_columns,
    ]
    _write_csv(out / "transcript_scores.csv", transcript_fields, transcript_rows)

    transcript_region_rows: list[dict[str, Any]] = []
    for outcome in result.region_outcomes:
        row = {
            "transcript_index": outcome.transcript_index,
            "transcript_id": result.sequence_ids[outcome.transcript_index],
            "region": outcome.region,
            "status": outcome.status,
            "skip_reason": outcome.skip_reason or "",
            **_statistics(result.scores[outcome.instance_start : outcome.instance_end]),
        }
        _add_metadata(row, result.metadata[outcome.transcript_index], metadata_columns)
        transcript_region_rows.append(row)
    transcript_region_fields = [
        "transcript_index",
        "transcript_id",
        "region",
        "status",
        "skip_reason",
        *statistic_fields,
        *metadata_columns,
    ]
    _write_csv(out / "transcript_region_scores.csv", transcript_region_fields, transcript_region_rows)

    skip_counts = Counter(
        outcome.skip_reason for outcome in result.region_outcomes if outcome.skip_reason is not None
    )
    transcript_status_counts = Counter(str(row["status"]) for row in transcript_rows)
    arrays = {
        "scores": {"shape": list(result.scores.shape), "dtype": str(result.scores.dtype)},
    }
    if result.sequences is not None:
        arrays["sequences"] = {
            "shape": list(result.sequences.shape),
            "dtype": str(result.sequences.dtype),
        }
    summary = {
        "analysis": "endogenous_legnet_window_scan",
        "score_definition": "raw checkpoint scalar prediction",
        "score_centering": None,
        "stability_direction": "depends_on_checkpoint_training_target",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset": str(dataset) if dataset is not None else None,
        "regions": list(result.regions),
        "window_size": result.window_size,
        "stride": result.stride,
        "default_stride_definition": "max(1, window_size // 4)",
        "coordinate_convention": "zero_based_half_open",
        "padding_policy": "right_pad_short_region_or_one_trailing_grid_window_with_all_zero_columns",
        "window_order": "transcript_then_requested_region_then_start",
        "input_shape": list(result.input_shape),
        "n_transcripts": len(result.sequence_ids),
        "n_windows": len(result.instances),
        "n_padded_windows": sum(instance.padding_length > 0 for instance in result.instances),
        "n_region_outcomes": len(result.region_outcomes),
        "transcript_status_counts": dict(sorted(transcript_status_counts.items())),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "arrays": arrays,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("scan-legnet: done", enabled=progress)


__all__ = [
    "LegNetScanResult",
    "LegNetWindowInstance",
    "RegionScanOutcome",
    "generate_scan_windows",
    "normalize_scan_regions",
    "save_legnet_scan_result",
    "scan_legnet_windows",
]
