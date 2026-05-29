from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.edits import scramble_motif_ablating_inplace
from transcriptml.interpret.motifs import find_motif_starts, parse_motif
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_result_dir
from transcriptml.interpret.codon_ism import find_cds_codon_starts
from transcriptml.progress import ProgressReporter, log_progress


@dataclass(frozen=True)
class MotifInstance:
    instance_index: int
    seq_index: int
    motif: str
    start: int
    end: int
    valid_length: int
    region: str | None = None


@dataclass
class MotifAblationResult:
    instances: list[MotifInstance]
    reference_predictions: np.ndarray
    ablation_predictions: np.ndarray
    effects: np.ndarray
    region: str | None = None


def normalize_motif_region(region: str | None) -> str | None:
    """Normalize optional region filters for motif analyses.

    Args:
        region: Optional user-supplied region label such as ``5utr``, ``cds``,
            ``3utr``, ``all``, or ``None``.
    """

    if region is None:
        return None
    key = str(region).strip().lower().replace("-", "").replace("_", "").replace("'", "")
    if not key or key in {"all", "none", "full", "transcript", "wholetranscript"}:
        return None
    if key in {"5utr", "utr5", "fiveutr", "fiveprimeutr"}:
        return "5utr"
    if key in {"cds", "coding", "codingsequence"}:
        return "cds"
    if key in {"3utr", "utr3", "threeutr", "threeprimeutr"}:
        return "3utr"
    raise ValueError("region must be one of: 5utr, cds, 3utr")


def motif_region_bounds(
    x: np.ndarray,
    *,
    region: str | None,
    valid_length: int,
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
) -> tuple[int, int] | None:
    """Return half-open transcript-coordinate bounds for a requested region.

    ``None`` means a requested region cannot be found because the sequence lacks
    a CDS annotation or usable CDS channel.

    Args:
        x: Encoded ``(C, L)`` transcript array.
        region: Optional normalized or raw region label to bound.
        valid_length: Valid transcript length within ``x``.
        schema: Sequence schema name or object describing channel layout.
        cds_channel: Optional CDS channel name or integer index. When omitted,
            the CDS channel is inferred from ``schema``.
    """

    normalized = normalize_motif_region(region)
    if normalized is None:
        return 0, int(valid_length)
    if np.asarray(x).shape[0] < 5:
        return None
    try:
        cds = find_cds_codon_starts(x, schema, valid_length=valid_length, cds_channel=cds_channel)
    except ValueError:
        return None
    if cds.cds_length < 3 or cds.starts.size == 0:
        return None
    cds_start = max(0, int(cds.cds_start))
    cds_end = min(int(valid_length), int(cds.cds_end) + 1)
    if normalized == "5utr":
        return 0, cds_start
    if normalized == "cds":
        return cds_start, cds_end
    return cds_end, int(valid_length)


def motif_site_in_region(start: int, end: int, bounds: tuple[int, int]) -> bool:
    """Return whether a motif interval is fully inside a half-open region.

    Args:
        start: Zero-based inclusive motif start coordinate.
        end: Zero-based exclusive motif end coordinate.
        bounds: Half-open region bounds as ``(start, end)``.
    """

    region_start, region_end = bounds
    return int(start) >= region_start and int(end) <= region_end


def enumerate_motif_instances(
    X: np.ndarray,
    motif: str,
    *,
    valid_lengths: Sequence[int] | None = None,
    region: str | None = None,
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
    progress: bool = True,
) -> list[MotifInstance]:
    """Find all motif instances in a batch of encoded sequences.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with base channels first.
        motif: Motif string accepted by ``parse_motif``.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred from ``X``.
        region: Optional region filter limiting motif sites to ``5utr``,
            ``cds``, or ``3utr``.
        schema: Sequence schema name or object used for region-aware scans.
        cds_channel: Optional CDS channel name or integer index for region
            filtering.
        progress: Whether to emit progress messages while scanning.
    """

    arr = np.asarray(X)
    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    motif_sets = parse_motif(motif)
    normalized_region = normalize_motif_region(region)
    out: list[MotifInstance] = []
    reporter = ProgressReporter(
        f"motif scan '{motif}'",
        total=int(arr.shape[0]),
        unit="sequences",
        enabled=progress,
    )
    for seq_i in range(arr.shape[0]):
        valid_len = min(int(lengths[seq_i]), int(arr.shape[-1]))
        bounds = motif_region_bounds(
            arr[seq_i],
            region=normalized_region,
            valid_length=valid_len,
            schema=schema,
            cds_channel=cds_channel,
        )
        if bounds is None:
            reporter.update()
            continue
        starts = find_motif_starts(arr[seq_i, :4, :valid_len], motif_sets)
        for start in starts.tolist():
            end = int(start + len(motif_sets))
            if not motif_site_in_region(int(start), end, bounds):
                continue
            out.append(
                MotifInstance(
                    instance_index=len(out),
                    seq_index=int(seq_i),
                    motif=motif,
                    start=int(start),
                    end=end,
                    valid_length=valid_len,
                    region=normalized_region,
                )
            )
        reporter.update()
    reporter.close(extra=f"{len(out)} instances")
    return out


def mean_ablation_prediction(
    x_ref: np.ndarray,
    predictor: Predictor,
    *,
    motif_start: int,
    motif_sets: Sequence[set[int]],
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
) -> float:
    """Predict the mean response after repeated motif-scrambling ablations.

    Args:
        x_ref: Reference encoded ``(C, L)`` sequence.
        predictor: Predictor used to score scrambled sequences.
        motif_start: Zero-based motif start position in ``x_ref``.
        motif_sets: Parsed motif position sets from ``parse_motif``.
        n_scrambles: Number of independently scrambled ablations to average.
        strategy: Scrambling strategy name supported by the edits module.
        rng: NumPy random generator used for reproducible scrambling.
    """

    if n_scrambles <= 0:
        return float(predictor.predict(x_ref[None, :, :])[0])
    batch = np.repeat(x_ref[None, :, :], int(n_scrambles), axis=0).copy()
    for i in range(batch.shape[0]):
        scramble_motif_ablating_inplace(
            batch[i],
            motif_start=motif_start,
            motif_sets=motif_sets,
            strategy=strategy,
            rng=rng,
        )
    return float(predictor.predict(batch).mean(dtype=np.float64))


def motif_ablation(
    X: np.ndarray,
    predictor: Predictor,
    *,
    motif: str,
    n_scrambles: int = 10,
    strategy: str = "random_different",
    seed: int = 123,
    valid_lengths: Sequence[int] | None = None,
    region: str | None = None,
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
    progress: bool = True,
) -> MotifAblationResult:
    """Compute motif ablation effect ``A - R`` for each motif instance.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with base channels first.
        predictor: Predictor used to score reference and ablated sequences.
        motif: Motif string accepted by ``parse_motif``.
        n_scrambles: Number of scrambled ablations to average per motif
            instance.
        strategy: Scrambling strategy name supported by the edits module.
        seed: Random seed used for ablation scrambling.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred from ``X``.
        region: Optional region filter limiting motif sites to ``5utr``,
            ``cds``, or ``3utr``.
        schema: Sequence schema name or object used for region-aware scans.
        cds_channel: Optional CDS channel name or integer index for region
            filtering.
        progress: Whether to emit progress messages while running the scan.
    """

    arr = np.asarray(X)
    motif_sets = parse_motif(motif)
    normalized_region = normalize_motif_region(region)
    instances = enumerate_motif_instances(
        arr,
        motif,
        valid_lengths=valid_lengths,
        region=normalized_region,
        schema=schema,
        cds_channel=cds_channel,
        progress=progress,
    )
    log_progress(f"motif-ablation: predicting {arr.shape[0]} reference sequences", enabled=progress)
    ref_by_seq = predictor.predict(arr)
    R = np.zeros(len(instances), dtype=np.float32)
    A = np.zeros(len(instances), dtype=np.float32)
    rng = np.random.default_rng(seed)
    reporter = ProgressReporter(
        "motif-ablation: ablate instances",
        total=len(instances),
        unit="instances",
        enabled=progress,
    )
    for inst in instances:
        x_ref = arr[inst.seq_index]
        R[inst.instance_index] = ref_by_seq[inst.seq_index]
        A[inst.instance_index] = mean_ablation_prediction(
            x_ref,
            predictor,
            motif_start=inst.start,
            motif_sets=motif_sets,
            n_scrambles=n_scrambles,
            strategy=strategy,
            rng=rng,
        )
        reporter.update()
    reporter.close()
    return MotifAblationResult(
        instances=instances,
        reference_predictions=R,
        ablation_predictions=A,
        effects=A - R,
        region=normalized_region,
    )


def save_motif_ablation_result(result: MotifAblationResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save motif ablation arrays, instance table, and summary metadata.

    Args:
        result: Motif ablation result object to serialize.
        out_dir: Destination directory for arrays, tables, and summary JSON.
        progress: Whether to emit progress messages while saving.
    """

    log_progress(f"motif-ablation: saving results to {out_dir}", enabled=progress)
    save_result_dir(
        out_dir,
        arrays={
            "reference_predictions": result.reference_predictions,
            "ablation_predictions": result.ablation_predictions,
            "effects": result.effects,
        },
        tables={"instances": result.instances},
        summary={
            "analysis": "motif_ablation",
            "effect_definition": "A - R",
            "n_instances": len(result.instances),
            "region": result.region,
        },
    )
    log_progress("motif-ablation: done", enabled=progress)
