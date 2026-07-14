from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.ablation import (
    MotifInstance,
    enumerate_motif_instances,
    mean_ablation_prediction,
    normalize_motif_region,
)
from transcriptml.interpret.edits import scramble_motif_ablating_inplace, scramble_window_inplace, valid_base_window
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
        progress: Whether to emit progress messages while scanning.
    """

    arr = np.asarray(X)
    motif_sets = parse_motif(motif)
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
    log_progress(f"motif-context: predicting {arr.shape[0]} reference sequences", enabled=progress)
    ref_by_seq = predictor.predict(arr)
    rng = np.random.default_rng(seed)
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
    for inst in instances:
        x_ref = arr[inst.seq_index]
        R[inst.instance_index] = ref_by_seq[inst.seq_index]
        A[inst.instance_index] = mean_ablation_prediction(
            x_ref,
            predictor,
            motif_start=inst.start,
            motif_sets=motif_sets,
            n_scrambles=n_motif_scrambles,
            strategy=strategy,
            rng=rng,
        )
        ablation_ref = float(A[inst.instance_index] - R[inst.instance_index])
        if window_size <= 0 or n_window_scrambles <= 0:
            instance_reporter.update()
            continue
        max_start = int(inst.valid_length) - int(window_size)
        if max_start < 0:
            instance_reporter.update()
            continue
        width = max(L, inst.valid_length) if context_width is None else int(context_width)
        start_min = max(0, inst.start - width)
        start_max = min(max_start, inst.end - 1 + width)
        for w_start in range(start_min, start_max + 1):
            w_end = w_start + int(window_size)
            if intervals_overlap(w_start, w_end, inst.start, inst.end):
                continue
            if not valid_base_window(x_ref, w_start, w_end):
                continue
            deltas: list[float] = []
            for _ in range(int(n_window_scrambles)):
                x_ctx = x_ref.copy()
                scramble_window_inplace(
                    x_ctx,
                    start=w_start,
                    window_size=int(window_size),
                    strategy=strategy,
                    rng=rng,
                )
                M = float(predictor.predict(x_ctx[None, :, :])[0])
                if n_motif_scrambles <= 0:
                    MA = M
                else:
                    batch = np.repeat(x_ctx[None, :, :], int(n_motif_scrambles), axis=0).copy()
                    for b in range(batch.shape[0]):
                        scramble_motif_ablating_inplace(
                            batch[b],
                            motif_start=inst.start,
                            motif_sets=motif_sets,
                            strategy=strategy,
                            rng=rng,
                        )
                    MA = float(predictor.predict(batch).mean(dtype=np.float64))
                deltas.append((MA - M) - ablation_ref)
            context_effects[inst.instance_index, w_start] = float(np.mean(deltas))
            context_mask[inst.instance_index, w_start] = 1
            window_reporter.update()
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
        },
    )
    log_progress("motif-context: done", enabled=progress)
