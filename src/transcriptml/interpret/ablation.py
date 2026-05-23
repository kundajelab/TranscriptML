from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.interpret.edits import scramble_motif_ablating_inplace
from transcriptml.interpret.motifs import find_motif_starts, parse_motif
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.results import save_result_dir


@dataclass(frozen=True)
class MotifInstance:
    instance_index: int
    seq_index: int
    motif: str
    start: int
    end: int
    valid_length: int


@dataclass
class MotifAblationResult:
    instances: list[MotifInstance]
    reference_predictions: np.ndarray
    ablation_predictions: np.ndarray
    effects: np.ndarray


def enumerate_motif_instances(
    X: np.ndarray,
    motif: str,
    *,
    valid_lengths: Sequence[int] | None = None,
) -> list[MotifInstance]:
    """Find all motif instances in a batch of encoded sequences."""

    arr = np.asarray(X)
    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    motif_sets = parse_motif(motif)
    out: list[MotifInstance] = []
    for seq_i in range(arr.shape[0]):
        valid_len = min(int(lengths[seq_i]), int(arr.shape[-1]))
        starts = find_motif_starts(arr[seq_i, :4, :valid_len], motif_sets)
        for start in starts.tolist():
            out.append(
                MotifInstance(
                    instance_index=len(out),
                    seq_index=int(seq_i),
                    motif=motif,
                    start=int(start),
                    end=int(start + len(motif_sets)),
                    valid_length=valid_len,
                )
            )
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
    """Predict the mean response after repeated motif-scrambling ablations."""

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
) -> MotifAblationResult:
    """Compute motif ablation effect ``A - R`` for each motif instance."""

    arr = np.asarray(X)
    motif_sets = parse_motif(motif)
    instances = enumerate_motif_instances(arr, motif, valid_lengths=valid_lengths)
    ref_by_seq = predictor.predict(arr)
    R = np.zeros(len(instances), dtype=np.float32)
    A = np.zeros(len(instances), dtype=np.float32)
    rng = np.random.default_rng(seed)
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
    return MotifAblationResult(instances=instances, reference_predictions=R, ablation_predictions=A, effects=A - R)


def save_motif_ablation_result(result: MotifAblationResult, out_dir: str | Path) -> None:
    """Save motif ablation arrays, instance table, and summary metadata."""

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
        },
    )
