from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.interpret.predictor import Predictor
from transcriptml.progress import ProgressReporter, log_progress


@dataclass
class ISMResult:
    deltas: np.ndarray
    reference_predictions: np.ndarray
    valid_lengths: np.ndarray


def compute_ism(
    X: np.ndarray,
    predictor: Predictor,
    *,
    valid_lengths: np.ndarray | None = None,
    mutation_batch_size: int = 512,
    progress: bool = True,
) -> ISMResult:
    """Single-nucleotide ISM with signed mutant-minus-reference effects.

    The returned ``deltas`` array has shape ``(N, 4, L)``. At each valid base
    position, the three alternative base channels store mutant-reference
    deltas; the original-base channel remains zero.
    """

    arr = np.asarray(X)
    if arr.ndim != 3 or arr.shape[1] < 4:
        raise ValueError(f"Expected X with shape (N, C>=4, L), got {arr.shape}")
    N, _, L = arr.shape
    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    log_progress(f"ism: predicting {N} reference sequences", enabled=progress)
    ref = predictor.predict(arr)
    deltas = np.zeros((N, 4, L), dtype=np.float32)
    batch: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    n_mutants = 0

    def flush() -> None:
        """Predict and store any queued mutant sequences."""

        if not batch:
            return
        preds = predictor.predict(np.stack(batch, axis=0))
        for pred, (seq_i, base_i, pos_i) in zip(preds, meta):
            deltas[seq_i, base_i, pos_i] = float(pred - ref[seq_i])
        batch.clear()
        meta.clear()

    reporter = ProgressReporter("ism: scan sequences", total=N, unit="sequences", enabled=progress)
    for i in range(N):
        seq_len = min(int(lengths[i]), L)
        for pos in range(seq_len):
            col = arr[i, :4, pos]
            if np.count_nonzero(col) != 1:
                continue
            orig = int(np.argmax(col))
            for new_base in range(4):
                if new_base == orig:
                    continue
                mut = arr[i].copy()
                mut[:4, pos] = 0
                mut[new_base, pos] = 1
                batch.append(mut)
                meta.append((i, new_base, pos))
                n_mutants += 1
                if len(batch) >= mutation_batch_size:
                    flush()
        reporter.update()
    flush()
    reporter.close(extra=f"{n_mutants} mutants predicted")
    return ISMResult(deltas=deltas, reference_predictions=ref.astype(np.float32), valid_lengths=lengths)


def max_abs_effect_per_position(deltas: np.ndarray) -> np.ndarray:
    """Summarize ISM effects by maximum absolute base substitution effect."""

    return np.max(np.abs(np.asarray(deltas)), axis=1)


def save_ism_result(result: ISMResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save ISM result arrays and a small JSON summary."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"ism: saving results to {out}", enabled=progress)
    np.save(out / "deltas.npy", result.deltas)
    np.save(out / "reference_predictions.npy", result.reference_predictions)
    np.save(out / "valid_lengths.npy", result.valid_lengths)
    np.save(out / "max_abs_effect.npy", max_abs_effect_per_position(result.deltas))
    summary = {
        "analysis": "single_nucleotide_ism",
        "delta_definition": "mutant_prediction - reference_prediction",
        "deltas_shape": list(result.deltas.shape),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("ism: done", enabled=progress)
