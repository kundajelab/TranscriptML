from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.interpret.ablation import MotifInstance, enumerate_motif_instances, mean_ablation_prediction
from transcriptml.interpret.edits import scramble_motif_ablating_inplace, scramble_window_inplace, valid_base_window
from transcriptml.interpret.motifs import intervals_overlap, parse_motif
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_result_dir


@dataclass
class MotifContextResult:
    instances: list[MotifInstance]
    ablation_effects: np.ndarray
    context_effects: np.ndarray
    context_mask: np.ndarray
    reference_predictions: np.ndarray
    ablation_predictions: np.ndarray


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
) -> MotifContextResult:
    """Scan context windows with effect ``(MA - M) - (A - R)``."""

    arr = np.asarray(X)
    motif_sets = parse_motif(motif)
    instances = enumerate_motif_instances(arr, motif, valid_lengths=valid_lengths)
    L = int(arr.shape[-1])
    context_effects = np.zeros((len(instances), L), dtype=np.float32)
    context_mask = np.zeros((len(instances), L), dtype=np.uint8)
    R = np.zeros(len(instances), dtype=np.float32)
    A = np.zeros(len(instances), dtype=np.float32)
    ref_by_seq = predictor.predict(arr)
    rng = np.random.default_rng(seed)
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
            continue
        max_start = int(inst.valid_length) - int(window_size)
        if max_start < 0:
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
    return MotifContextResult(
        instances=instances,
        ablation_effects=A - R,
        context_effects=context_effects,
        context_mask=context_mask,
        reference_predictions=R,
        ablation_predictions=A,
    )


def save_motif_context_result(result: MotifContextResult, out_dir: str | Path) -> None:
    """Save motif context scan arrays, instance table, and summary metadata."""

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
        },
    )
