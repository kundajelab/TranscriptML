from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.ablation import (
    mean_ablation_prediction,
    motif_region_bounds,
    motif_site_in_region,
    normalize_motif_region,
)
from transcriptml.interpret.edits import scramble_motif_ablating_inplace
from transcriptml.interpret.motifs import find_motif_starts, intervals_overlap, parse_motif
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_result_dir
from transcriptml.progress import ProgressReporter, log_progress


@dataclass(frozen=True)
class Site:
    label: str
    start: int
    end: int
    motif: str


@dataclass(frozen=True)
class PairRecord:
    pair_index: int
    seq_index: int
    valid_length: int
    site1_label: str
    site1_start: int
    site1_end: int
    site2_label: str
    site2_start: int
    site2_end: int
    region: str | None = None


@dataclass
class EpistasisResult:
    pairs: list[PairRecord]
    reference_predictions: np.ndarray
    single_ablation_predictions: np.ndarray
    paired_ablation_predictions: np.ndarray
    single_ablation_effects: np.ndarray
    paired_ablation_effects: np.ndarray
    epistasis: np.ndarray
    region: str | None = None


def _enumerate_pairs_for_sequence(
    x: np.ndarray,
    *,
    seq_index: int,
    valid_length: int,
    motif: str,
    motif_sets: Sequence[set[int]],
    motif2: str | None,
    motif2_sets: Sequence[set[int]] | None,
    skip_overlaps: bool,
    region_bounds: tuple[int, int] | None = None,
) -> list[tuple[Site, Site]]:
    """Enumerate candidate motif-site pairs for one encoded sequence."""

    starts1 = find_motif_starts(x[:4, :valid_length], motif_sets)
    len1 = len(motif_sets)
    sites1 = []
    for start_raw in starts1:
        start = int(start_raw)
        end = int(start + len1)
        if region_bounds is not None and not motif_site_in_region(start, end, region_bounds):
            continue
        sites1.append(Site("motif1" if motif2 else "motif", start, end, motif))
    pairs: list[tuple[Site, Site]] = []
    if motif2 is None:
        for i in range(len(sites1)):
            for j in range(i + 1, len(sites1)):
                a, b = sites1[i], sites1[j]
                if skip_overlaps and intervals_overlap(a.start, a.end, b.start, b.end):
                    continue
                pairs.append((a, b))
        return pairs
    assert motif2_sets is not None
    starts2 = find_motif_starts(x[:4, :valid_length], motif2_sets)
    len2 = len(motif2_sets)
    sites2 = []
    for start_raw in starts2:
        start = int(start_raw)
        end = int(start + len2)
        if region_bounds is not None and not motif_site_in_region(start, end, region_bounds):
            continue
        sites2.append(Site("motif2", start, end, motif2))
    for a in sites1:
        for b in sites2:
            if skip_overlaps and intervals_overlap(a.start, a.end, b.start, b.end):
                continue
            pairs.append((a, b))
    return pairs


def _mean_multi_ablation_prediction(
    x_ref: np.ndarray,
    predictor: Predictor,
    *,
    edits: Sequence[tuple[int, Sequence[set[int]]]],
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
) -> float:
    """Predict the mean response after applying multiple motif ablations."""

    if n_scrambles <= 0:
        return float(predictor.predict(x_ref[None, :, :])[0])
    batch = np.repeat(x_ref[None, :, :], int(n_scrambles), axis=0).copy()
    for b in range(batch.shape[0]):
        for start, motif_sets in edits:
            scramble_motif_ablating_inplace(
                batch[b],
                motif_start=start,
                motif_sets=motif_sets,
                strategy=strategy,
                rng=rng,
            )
    return float(predictor.predict(batch).mean(dtype=np.float64))


def motif_epistasis(
    X: np.ndarray,
    predictor: Predictor,
    *,
    motif: str,
    motif2: str | None = None,
    n_scrambles: int = 10,
    strategy: str = "random_different",
    seed: int = 123,
    skip_overlaps: bool = True,
    max_pairs: int | None = None,
    valid_lengths: Sequence[int] | None = None,
    region: str | None = None,
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
    progress: bool = True,
) -> EpistasisResult:
    """Compute pairwise epistasis ``A12 - A1 - A2 + R``."""

    arr = np.asarray(X)
    motif_sets = parse_motif(motif)
    motif2_sets = parse_motif(motif2) if motif2 is not None else None
    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    normalized_region = normalize_motif_region(region)
    log_progress(f"epistasis: predicting {arr.shape[0]} reference sequences", enabled=progress)
    ref_by_seq = predictor.predict(arr)
    rng = np.random.default_rng(seed)
    records: list[PairRecord] = []
    site_sets: dict[tuple[int, str, int, int], Sequence[set[int]]] = {}
    raw_pairs: list[tuple[int, Site, Site]] = []
    enumerate_reporter = ProgressReporter(
        "epistasis: enumerate motif pairs",
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
            enumerate_reporter.update()
            continue
        pairs = _enumerate_pairs_for_sequence(
            arr[seq_i],
            seq_index=seq_i,
            valid_length=valid_len,
            motif=motif,
            motif_sets=motif_sets,
            motif2=motif2,
            motif2_sets=motif2_sets,
            skip_overlaps=skip_overlaps,
            region_bounds=bounds,
        )
        for site1, site2 in pairs:
            if max_pairs is not None and len(raw_pairs) >= int(max_pairs):
                break
            raw_pairs.append((seq_i, site1, site2))
            site_sets[(seq_i, site1.motif, site1.start, site1.end)] = (
                motif_sets if site1.motif == motif else motif2_sets
            )
            site_sets[(seq_i, site2.motif, site2.start, site2.end)] = (
                motif_sets if site2.motif == motif else motif2_sets
            )
        if max_pairs is not None and len(raw_pairs) >= int(max_pairs):
            enumerate_reporter.update()
            break
        enumerate_reporter.update()
    enumerate_reporter.close(extra=f"{len(raw_pairs)} pairs")
    P = len(raw_pairs)
    R = np.zeros(P, dtype=np.float32)
    singles = np.zeros((P, 2), dtype=np.float32)
    paired = np.zeros(P, dtype=np.float32)
    single_cache: dict[tuple[int, str, int, int], float] = {}
    pair_reporter = ProgressReporter("epistasis: score pairs", total=P, unit="pairs", enabled=progress)
    for pair_i, (seq_i, site1, site2) in enumerate(raw_pairs):
        valid_len = min(int(lengths[seq_i]), int(arr.shape[-1]))
        records.append(
            PairRecord(
                pair_index=pair_i,
                seq_index=int(seq_i),
                valid_length=valid_len,
                site1_label=site1.label,
                site1_start=site1.start,
                site1_end=site1.end,
                site2_label=site2.label,
                site2_start=site2.start,
                site2_end=site2.end,
                region=normalized_region,
            )
        )
        x_ref = arr[seq_i]
        R[pair_i] = ref_by_seq[seq_i]
        for col, site in enumerate((site1, site2)):
            key = (seq_i, site.motif, site.start, site.end)
            if key not in single_cache:
                sets = site_sets[key]
                if sets is None:
                    raise ValueError("Missing motif sets for site")
                single_cache[key] = mean_ablation_prediction(
                    x_ref,
                    predictor,
                    motif_start=site.start,
                    motif_sets=sets,
                    n_scrambles=n_scrambles,
                    strategy=strategy,
                    rng=rng,
                )
            singles[pair_i, col] = single_cache[key]
        edits = [
            (site1.start, site_sets[(seq_i, site1.motif, site1.start, site1.end)]),
            (site2.start, site_sets[(seq_i, site2.motif, site2.start, site2.end)]),
        ]
        paired[pair_i] = _mean_multi_ablation_prediction(
            x_ref,
            predictor,
            edits=edits,
            n_scrambles=n_scrambles,
            strategy=strategy,
            rng=rng,
        )
        pair_reporter.update()
    pair_reporter.close(extra=f"{len(single_cache)} unique single sites")
    single_effects = singles - R[:, None]
    paired_effects = paired - R
    epi = paired - singles[:, 0] - singles[:, 1] + R
    return EpistasisResult(
        pairs=records,
        reference_predictions=R,
        single_ablation_predictions=singles,
        paired_ablation_predictions=paired,
        single_ablation_effects=single_effects,
        paired_ablation_effects=paired_effects,
        epistasis=epi.astype(np.float32),
        region=normalized_region,
    )


def save_epistasis_result(result: EpistasisResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save epistasis arrays, pair table, and summary metadata."""

    log_progress(f"epistasis: saving results to {out_dir}", enabled=progress)
    save_result_dir(
        out_dir,
        arrays={
            "reference_predictions": result.reference_predictions,
            "single_ablation_predictions": result.single_ablation_predictions,
            "single_ablation_effects": result.single_ablation_effects,
            "paired_ablation_predictions": result.paired_ablation_predictions,
            "paired_ablation_effects": result.paired_ablation_effects,
            "epistasis": result.epistasis,
        },
        tables={"pairs": result.pairs},
        summary={
            "analysis": "motif_epistasis",
            "epistasis_definition": "A12 - A1 - A2 + R",
            "n_pairs": len(result.pairs),
            "region": result.region,
        },
    )
    log_progress("epistasis: done", enabled=progress)
