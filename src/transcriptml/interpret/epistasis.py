from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.ablation import (
    _build_motif_ablated_batch,
    motif_region_bounds,
    motif_site_in_region,
    normalize_motif_region,
)
from transcriptml.interpret.backends import normalize_backend
from transcriptml.interpret.edits import (
    effective_perturbation_backend,
    extract_base_indices,
    scramble_motif_ablating_batch,
)
from transcriptml.interpret.motifs import (
    find_motif_starts,
    intervals_overlap,
    parse_motif,
)
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
    backend: str = "tangermeme"
    strategy: str = "random_different"


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
    """Enumerate candidate motif-site pairs for one encoded sequence.

    Args:
        x: Encoded ``(C, L)`` sequence with base channels first.
        seq_index: Index of the sequence within the source batch.
        valid_length: Valid transcript length to scan within ``x``.
        motif: Primary motif string.
        motif_sets: Parsed primary motif position sets.
        motif2: Optional secondary motif string. When omitted, primary motif
            sites are paired with each other.
        motif2_sets: Parsed secondary motif position sets, required when
            ``motif2`` is provided.
        skip_overlaps: Whether to exclude overlapping site pairs.
        region_bounds: Optional half-open transcript-coordinate bounds that
            both sites must fall inside.
    """

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


def _build_multi_ablated_batch(
    x_ref: np.ndarray,
    *,
    edits: Sequence[tuple[int, Sequence[set[int]]]],
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
    backend: str,
) -> np.ndarray:
    """Create ``(n_scrambles, C, L)`` copies with multiple motif sites ablated."""

    count = int(n_scrambles)
    batch = np.repeat(np.asarray(x_ref)[None, :, :], count, axis=0)
    rows = np.arange(count)[:, None]
    for start_raw, motif_sets in edits:
        start = int(start_raw)
        k = len(motif_sets)
        # Read from batch[0] before writing when sites are disjoint, or per-row
        # if overlapping sites were explicitly requested.
        orig = extract_base_indices(x_ref, start, start + k)
        cands = scramble_motif_ablating_batch(
            orig,
            motif_sets=motif_sets,
            n_scrambles=count,
            strategy=strategy,
            rng=rng,
            backend=backend,
        )
        batch[:, :4, start : start + k] = 0
        batch[rows, cands, start + np.arange(k, dtype=np.int64)[None, :]] = 1
    return batch


def _mean_multi_ablation_prediction(
    x_ref: np.ndarray,
    predictor: Predictor,
    *,
    edits: Sequence[tuple[int, Sequence[set[int]]]],
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
    backend: str = "tangermeme",
) -> float:
    """Predict the mean response after applying multiple motif ablations.

    Args:
        x_ref: Reference encoded ``(C, L)`` sequence.
        predictor: Predictor used to score multi-ablated sequences.
        edits: Sequence of ``(motif_start, motif_sets)`` ablations to apply to
            each mutant copy.
        n_scrambles: Number of independently scrambled multi-ablations to
            average.
        strategy: Scrambling strategy name supported by the edits module.
        rng: NumPy random generator used for reproducible scrambling.
        backend: Perturbation backend.
    """

    if n_scrambles <= 0:
        return float(predictor.predict(x_ref[None, :, :])[0])
    batch = _build_multi_ablated_batch(
        x_ref,
        edits=edits,
        n_scrambles=int(n_scrambles),
        strategy=strategy,
        rng=rng,
        backend=backend,
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
    mutation_batch_size: int = 512,
    backend: str = "tangermeme",
    progress: bool = True,
) -> EpistasisResult:
    """Compute pairwise epistasis ``A12 - A1 - A2 + R``.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with base channels first.
        predictor: Predictor used to score reference and ablated sequences.
        motif: Primary motif string accepted by ``parse_motif``.
        motif2: Optional secondary motif string. When omitted, pairs are formed
            among sites of ``motif``.
        n_scrambles: Number of scrambled ablations to average for each single
            or paired ablation.
        strategy: Scrambling strategy name supported by the edits module.
        seed: Random seed used for ablation scrambling.
        skip_overlaps: Whether to exclude overlapping motif-site pairs.
        max_pairs: Optional cap on the number of pairs to score.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred from ``X``.
        region: Optional region filter limiting motif sites to ``5utr``,
            ``cds``, or ``3utr``.
        schema: Sequence schema name or object used for region-aware scans.
        cds_channel: Optional CDS channel name or integer index for region
            filtering.
        mutation_batch_size: Maximum number of mutant sequences queued per
            prediction batch.
        backend: Perturbation backend.
        progress: Whether to emit progress messages while running the analysis.
    """

    if int(mutation_batch_size) <= 0:
        raise ValueError("mutation_batch_size must be positive")
    arr = np.asarray(X)
    selected_backend = normalize_backend(backend)
    motif_sets = parse_motif(motif)
    motif2_sets = parse_motif(motif2) if motif2 is not None else None
    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    normalized_region = normalize_motif_region(region)
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

    ref_by_seq: dict[int, float] = {}
    if P > 0:
        unique_seqs = np.unique(np.asarray([seq_i for seq_i, _, _ in raw_pairs], dtype=np.int64))
        log_progress(
            f"epistasis: predicting {unique_seqs.size} reference sequences",
            enabled=progress,
        )
        ref_preds = predictor.predict(arr[unique_seqs])
        ref_by_seq = {int(si): float(p) for si, p in zip(unique_seqs.tolist(), ref_preds.tolist())}

    unique_keys: list[tuple[int, str, int, int]] = []
    seen_keys: set[tuple[int, str, int, int]] = set()
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
        R[pair_i] = ref_by_seq[seq_i]
        for site in (site1, site2):
            key = (seq_i, site.motif, site.start, site.end)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_keys.append(key)

    count = int(n_scrambles)
    batch_limit = int(mutation_batch_size)
    single_cache: dict[tuple[int, str, int, int], float] = {}
    pair_reporter = ProgressReporter("epistasis: score pairs", total=P, unit="pairs", enabled=progress)

    if count <= 0:
        for key in unique_keys:
            single_cache[key] = ref_by_seq[key[0]]
        for pair_i, (seq_i, site1, site2) in enumerate(raw_pairs):
            singles[pair_i, 0] = single_cache[(seq_i, site1.motif, site1.start, site1.end)]
            singles[pair_i, 1] = single_cache[(seq_i, site2.motif, site2.start, site2.end)]
            paired[pair_i] = R[pair_i]
            pair_reporter.update()
    elif P > 0:
        single_sums = np.zeros(len(unique_keys), dtype=np.float64)
        chunks: list[np.ndarray] = []
        slices: list[tuple[int, int]] = []
        queued = 0

        def flush_into(target_sums: np.ndarray) -> None:
            nonlocal queued
            if not chunks:
                return
            batch_arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
            preds = predictor.predict(batch_arr)
            off = 0
            for idx, take in slices:
                target_sums[idx] += float(preds[off : off + take].sum(dtype=np.float64))
                off += take
            chunks.clear()
            slices.clear()
            queued = 0

        for site_idx, key in enumerate(unique_keys):
            sets = site_sets[key]
            if sets is None:
                raise ValueError("Missing motif sets for site")
            muts = _build_motif_ablated_batch(
                arr[key[0]],
                motif_start=key[2],
                motif_sets=sets,
                n_scrambles=count,
                strategy=strategy,
                rng=rng,
                backend=selected_backend,
            )
            cur = 0
            while cur < count:
                take = min(count - cur, batch_limit - queued)
                chunks.append(muts[cur : cur + take])
                slices.append((site_idx, take))
                queued += take
                cur += take
                if queued >= batch_limit:
                    flush_into(single_sums)
        flush_into(single_sums)
        for site_idx, key in enumerate(unique_keys):
            single_cache[key] = float(single_sums[site_idx] / count)

        paired_sums = np.zeros(P, dtype=np.float64)
        for pair_i, (seq_i, site1, site2) in enumerate(raw_pairs):
            singles[pair_i, 0] = single_cache[(seq_i, site1.motif, site1.start, site1.end)]
            singles[pair_i, 1] = single_cache[(seq_i, site2.motif, site2.start, site2.end)]
            edits = [
                (site1.start, site_sets[(seq_i, site1.motif, site1.start, site1.end)]),
                (site2.start, site_sets[(seq_i, site2.motif, site2.start, site2.end)]),
            ]
            muts_pair = _build_multi_ablated_batch(
                arr[seq_i],
                edits=edits,
                n_scrambles=count,
                strategy=strategy,
                rng=rng,
                backend=selected_backend,
            )
            cur = 0
            while cur < count:
                take = min(count - cur, batch_limit - queued)
                chunks.append(muts_pair[cur : cur + take])
                slices.append((pair_i, take))
                queued += take
                cur += take
                if queued >= batch_limit:
                    flush_into(paired_sums)
            pair_reporter.update()
        flush_into(paired_sums)
        paired[:] = (paired_sums / count).astype(np.float32)

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
        backend=selected_backend,
        strategy=strategy,
    )


def save_epistasis_result(result: EpistasisResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save epistasis arrays, pair table, and summary metadata.

    Args:
        result: Epistasis result object to serialize.
        out_dir: Destination directory for arrays, tables, and summary JSON.
        progress: Whether to emit progress messages while saving.
    """

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
            "backend": result.backend,
            "perturbation_backend": effective_perturbation_backend(result.strategy, result.backend),
            "strategy": result.strategy,
        },
    )
    log_progress("epistasis: done", enabled=progress)
