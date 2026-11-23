from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tangermeme.utils import random_one_hot

from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.ablation import (
    MotifInstance,
    _build_motif_ablated_batch,
    enumerate_motif_instances,
    normalize_motif_region,
)
from transcriptml.interpret.backends import normalize_backend
from transcriptml.interpret.edits import (
    effective_perturbation_backend,
    extract_base_indices,
    scramble_motif_ablating_batch,
    scramble_window_batch_ohe,
    valid_base_window,
)
from transcriptml.interpret.motifs import intervals_overlap, parse_motif
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_result_dir
from transcriptml.progress import ProgressReporter, log_progress


@dataclass
class MotifContextResult:
    instances: list[MotifInstance]
    ablation_effects: np.ndarray
    context_effects: np.ndarray
    context_mask: np.ndarray
    reference_predictions: np.ndarray
    ablation_predictions: np.ndarray
    region: str | None = None
    backend: str = "tangermeme"
    strategy: str = "random_different"


def motif_context_scan(
    X: np.ndarray,
    predictor: Predictor,
    *,
    motif: str,
    window_size: int = 5,
    context_width: int | None = None,
    n_motif_scrambles: int = 10,
    n_window_scrambles: int = 5,
    strategy: str = "random_different",
    seed: int = 123,
    valid_lengths: Sequence[int] | None = None,
    region: str | None = None,
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
    mutation_batch_size: int = 512,
    backend: str = "tangermeme",
    progress: bool = True,
) -> MotifContextResult:
    """Scan context windows with effect ``(MA - M) - (A - R)``.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with base channels first.
        predictor: Predictor used to score reference, motif-ablated, and
            context-scrambled sequences.
        motif: Motif string accepted by ``parse_motif``.
        window_size: Width of each context window to scramble.
        context_width: Maximum distance from the motif to scan. When ``None``,
            the full sequence length is considered.
        n_motif_scrambles: Number of motif ablations to average per context
            estimate.
        n_window_scrambles: Number of context-window scrambles to average.
        strategy: Scrambling strategy name supported by the edits module.
        seed: Random seed used for motif and context scrambling.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred during motif enumeration.
        region: Optional region filter limiting motif sites to ``5utr``,
            ``cds``, or ``3utr``.
        schema: Sequence schema name or object used for region-aware scans.
        cds_channel: Optional CDS channel name or integer index for region
            filtering.
        mutation_batch_size: Maximum number of mutant sequences queued per
            prediction batch.
        backend: Perturbation backend.
        progress: Whether to emit progress messages while scanning.
    """

    if int(mutation_batch_size) <= 0:
        raise ValueError("mutation_batch_size must be positive")
    arr = np.asarray(X)
    selected_backend = normalize_backend(backend)
    motif_sets = parse_motif(motif)
    motif_len = len(motif_sets)
    motif_cols = np.arange(motif_len, dtype=np.int64)
    instances = enumerate_motif_instances(
        arr,
        motif,
        valid_lengths=valid_lengths,
        region=region,
        schema=schema,
        cds_channel=cds_channel,
        progress=progress,
    )
    L = int(arr.shape[-1])
    context_effects = np.zeros((len(instances), L), dtype=np.float32)
    context_mask = np.zeros((len(instances), L), dtype=np.uint8)
    R = np.zeros(len(instances), dtype=np.float32)
    A = np.zeros(len(instances), dtype=np.float32)
    if instances:
        unique_seqs = np.unique(np.asarray([inst.seq_index for inst in instances], dtype=np.int64))
        log_progress(
            f"motif-context: predicting {unique_seqs.size} reference sequences",
            enabled=progress,
        )
        ref_preds = predictor.predict(arr[unique_seqs])
        ref_by_seq = {int(si): float(p) for si, p in zip(unique_seqs.tolist(), ref_preds.tolist())}
        for inst in instances:
            R[inst.instance_index] = ref_by_seq[inst.seq_index]

    rng = np.random.default_rng(seed)
    n_m_scrambles = int(n_motif_scrambles)
    n_w_scrambles = int(n_window_scrambles)
    w_width = int(window_size)
    batch_limit = int(mutation_batch_size)

    if n_m_scrambles <= 0:
        A[:] = R
    elif instances:
        a_sums = np.zeros(len(instances), dtype=np.float64)
        a_chunks: list[np.ndarray] = []
        a_slices: list[tuple[int, int]] = []
        a_queued = 0

        def flush_a() -> None:
            nonlocal a_queued
            if not a_chunks:
                return
            batch_arr = a_chunks[0] if len(a_chunks) == 1 else np.concatenate(a_chunks, axis=0)
            preds = predictor.predict(batch_arr)
            off = 0
            for idx, take in a_slices:
                a_sums[idx] += float(preds[off : off + take].sum(dtype=np.float64))
                off += take
            a_chunks.clear()
            a_slices.clear()
            a_queued = 0

        for inst in instances:
            muts_a = _build_motif_ablated_batch(
                arr[inst.seq_index],
                motif_start=inst.start,
                motif_sets=motif_sets,
                n_scrambles=n_m_scrambles,
                strategy=strategy,
                rng=rng,
                backend=selected_backend,
            )
            cur = 0
            while cur < n_m_scrambles:
                take = min(n_m_scrambles - cur, batch_limit - a_queued)
                a_chunks.append(muts_a[cur : cur + take])
                a_slices.append((inst.instance_index, take))
                a_queued += take
                cur += take
                if a_queued >= batch_limit:
                    flush_a()
        flush_a()
        A[:] = (a_sums / n_m_scrambles).astype(np.float32)

    instance_reporter = ProgressReporter(
        "motif-context: scan instances",
        total=len(instances),
        unit="instances",
        enabled=progress,
    )
    window_reporter = ProgressReporter(
        "motif-context: scan context windows",
        unit="windows",
        enabled=progress,
    )
    if w_width <= 0 or n_w_scrambles <= 0:
        for _ in instances:
            instance_reporter.update()
        window_reporter.close()
        instance_reporter.close()
        return MotifContextResult(
            instances=instances,
            ablation_effects=A - R,
            context_effects=context_effects,
            context_mask=context_mask,
            reference_predictions=R,
            ablation_predictions=A,
            region=normalize_motif_region(region),
            backend=selected_backend,
            strategy=strategy,
        )

    group_size = 1 + max(0, n_m_scrambles)
    reps_per_chunk = max(1, batch_limit // group_size)

    for inst in instances:
        x_ref = np.asarray(arr[inst.seq_index])
        ablation_ref = float(A[inst.instance_index] - R[inst.instance_index])
        max_start = int(inst.valid_length) - w_width
        if max_start < 0:
            instance_reporter.update()
            continue
        width = max(L, inst.valid_length) if context_width is None else int(context_width)
        start_min = max(0, inst.start - width)
        start_max = min(max_start, inst.end - 1 + width)
        valid_w_starts = [
            w_start
            for w_start in range(start_min, start_max + 1)
            if not intervals_overlap(w_start, w_start + w_width, inst.start, inst.end)
            and valid_base_window(x_ref, w_start, w_start + w_width)
        ]
        if not valid_w_starts:
            instance_reporter.update()
            continue

        n_w = len(valid_w_starts)
        if n_m_scrambles <= 0:
            for w_start in valid_w_starts:
                context_effects[inst.instance_index, w_start] = 0.0
                context_mask[inst.instance_index, w_start] = 1
            window_reporter.update(n_w)
            instance_reporter.update()
            continue

        total_w_reps = n_w * n_w_scrambles
        orig_motif = extract_base_indices(x_ref, inst.start, inst.end)
        if selected_backend == "tangermeme" and strategy == "random":
            w_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            all_w_ohe = (
                random_one_hot((total_w_reps, 4, w_width), dtype="int8", random_state=w_seed)
                .cpu()
                .numpy()
                .astype(arr.dtype, copy=False)
            )
        else:
            w_ohes = [
                scramble_window_batch_ohe(
                    extract_base_indices(x_ref, ws, ws + w_width),
                    n_scrambles=n_w_scrambles,
                    strategy=strategy,
                    rng=rng,
                    backend=selected_backend,
                    dtype=arr.dtype,
                )
                for ws in valid_w_starts
            ]
            all_w_ohe = np.concatenate(w_ohes, axis=0)

        rep_deltas = np.empty(total_w_reps, dtype=np.float64)
        for r0 in range(0, total_w_reps, reps_per_chunk):
            r1 = min(r0 + reps_per_chunk, total_w_reps)
            k_reps = r1 - r0
            buf = np.repeat(x_ref[None], k_reps * group_size, axis=0)
            # Draw all k_reps * n_m_scrambles motif ablations for this chunk in one batch!
            cands_chunk = scramble_motif_ablating_batch(
                orig_motif,
                motif_sets=motif_sets,
                n_scrambles=k_reps * n_m_scrambles,
                strategy=strategy,
                rng=rng,
                backend=selected_backend,
            )
            for j in range(k_reps):
                rep_idx = r0 + j
                ws = valid_w_starts[rep_idx // n_w_scrambles]
                g0 = j * group_size
                g1 = g0 + group_size
                buf[g0:g1, :4, ws : ws + w_width] = all_w_ohe[rep_idx]
                c_sub = cands_chunk[j * n_m_scrambles : (j + 1) * n_m_scrambles]
                buf[g0 + 1 : g1, :4, inst.start : inst.end] = 0
                buf[
                    np.arange(g0 + 1, g1)[:, None],
                    c_sub,
                    inst.start + motif_cols[None, :],
                ] = 1

            preds = predictor.predict(buf).reshape(k_reps, group_size)
            m_vals = preds[:, 0].astype(np.float64)
            ma_vals = preds[:, 1:].mean(axis=1, dtype=np.float64)
            rep_deltas[r0:r1] = (ma_vals - m_vals) - ablation_ref

        w_means = rep_deltas.reshape(n_w, n_w_scrambles).mean(axis=1)
        for ws, mean_val in zip(valid_w_starts, w_means.tolist()):
            context_effects[inst.instance_index, ws] = float(mean_val)
            context_mask[inst.instance_index, ws] = 1
        window_reporter.update(n_w)
        instance_reporter.update()

    window_reporter.close()
    instance_reporter.close()
    return MotifContextResult(
        instances=instances,
        ablation_effects=A - R,
        context_effects=context_effects,
        context_mask=context_mask,
        reference_predictions=R,
        ablation_predictions=A,
        region=normalize_motif_region(region),
        backend=selected_backend,
        strategy=strategy,
    )


def save_motif_context_result(result: MotifContextResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save motif context scan arrays, instance table, and summary metadata.

    Args:
        result: Motif context scan result object to serialize.
        out_dir: Destination directory for arrays, tables, and summary JSON.
        progress: Whether to emit progress messages while saving.
    """

    log_progress(f"motif-context: saving results to {out_dir}", enabled=progress)
    save_result_dir(
        out_dir,
        arrays={
            "reference_predictions": result.reference_predictions,
            "ablation_predictions": result.ablation_predictions,
            "ablation_effects": result.ablation_effects,
            "context_effects": result.context_effects,
            "context_mask": result.context_mask,
        },
        tables={"instances": result.instances},
        summary={
            "analysis": "motif_context",
            "context_effect_definition": "(MA - M) - (A - R)",
            "n_instances": len(result.instances),
            "region": result.region,
            "backend": result.backend,
            "perturbation_backend": effective_perturbation_backend(result.strategy, result.backend),
            "strategy": result.strategy,
        },
    )
    log_progress("motif-context: done", enabled=progress)
