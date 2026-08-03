from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.interpret.edits import scramble_window_inplace, valid_base_window
from transcriptml.interpret.predictor import Predictor
from transcriptml.progress import ProgressReporter, log_progress


@dataclass
class WindowISMResult:
    """Window-level random-mutagenesis effects for a sequence batch."""

    window_starts: np.ndarray
    window_mask: np.ndarray
    mean_deltas: np.ndarray
    mean_abs_deltas: np.ndarray
    std_deltas: np.ndarray
    reference_predictions: np.ndarray
    valid_lengths: np.ndarray
    input_shape: tuple[int, int, int]
    window_size: int
    stride: int
    n_ablations: int
    seed: int


def generate_window_starts(valid_length: int, window_size: int, stride: int) -> np.ndarray:
    """Generate fixed-width window starts, including a terminally anchored window.

    The final window ends exactly at ``valid_length``. Together with the
    requirement ``stride <= window_size``, this guarantees full base coverage
    for any sequence at least as long as the requested window.

    Args:
        valid_length: Number of represented sequence positions.
        window_size: Width of every window.
        stride: Distance between regular window starts.
    """

    length = int(valid_length)
    width = int(window_size)
    step = int(stride)
    if width <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0 or step > width:
        raise ValueError("stride must satisfy 1 <= stride <= window_size")
    if length < width:
        return np.empty((0,), dtype=np.int64)

    terminal_start = length - width
    starts = list(range(0, terminal_start + 1, step))
    if starts[-1] != terminal_start:
        starts.append(terminal_start)
    return np.asarray(starts, dtype=np.int64)


def _normalize_valid_lengths(X: np.ndarray, valid_lengths: Sequence[int] | None) -> np.ndarray:
    """Infer or validate one valid length per encoded sequence."""

    lengths = infer_valid_lengths(X) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    if lengths.shape != (X.shape[0],):
        raise ValueError(f"valid_lengths must have shape ({X.shape[0]},), got {lengths.shape}")
    if np.any(lengths < 0) or np.any(lengths > X.shape[-1]):
        raise ValueError(f"valid_lengths entries must be between 0 and encoded length {X.shape[-1]}")
    return lengths.astype(np.int64, copy=False)


def compute_window_ism(
    X: np.ndarray,
    predictor: Predictor,
    *,
    window_size: int,
    stride: int | None = None,
    n_ablations: int = 30,
    seed: int = 123,
    valid_lengths: Sequence[int] | None = None,
    mutation_batch_size: int = 512,
    progress: bool = True,
) -> WindowISMResult:
    """Compute repeated random-mutagenesis effects for fixed-width windows.

    Every nucleotide in a scored window is independently replaced by a
    uniformly sampled alternative base. Effects are signed mutant-minus-
    reference prediction differences. Replicate-level effects are summarized
    online as their mean, mean absolute value, and population standard
    deviation.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with at least four base
            channels.
        predictor: Predictor used to score reference and mutant sequences.
        window_size: Number of bases mutated in each window.
        stride: Distance between regular starts. Defaults to ``window_size``.
        n_ablations: Number of independently mutated sequences per window.
        seed: Non-negative base seed for deterministic per-window generators.
        valid_lengths: Optional represented length for each sequence.
        mutation_batch_size: Maximum number of mutants queued per prediction
            call.
        progress: Whether to emit progress messages.
    """

    arr = np.asarray(X)
    if arr.ndim != 3 or arr.shape[1] < 4:
        raise ValueError(f"Expected X with shape (N, C>=4, L), got {arr.shape}")
    width = int(window_size)
    step = width if stride is None else int(stride)
    if width <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0 or step > width:
        raise ValueError("stride must satisfy 1 <= stride <= window_size")
    if int(n_ablations) <= 0:
        raise ValueError("n_ablations must be positive")
    if int(mutation_batch_size) <= 0:
        raise ValueError("mutation_batch_size must be positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")

    n_sequences = int(arr.shape[0])
    lengths = _normalize_valid_lengths(arr, valid_lengths)
    starts_by_sequence = [generate_window_starts(int(length), width, step) for length in lengths]
    max_windows = max((len(starts) for starts in starts_by_sequence), default=0)
    starts_out = np.full((n_sequences, max_windows), -1, dtype=np.int64)
    mask = np.zeros((n_sequences, max_windows), dtype=bool)
    for seq_i, starts in enumerate(starts_by_sequence):
        starts_out[seq_i, : len(starts)] = starts
        for window_i, start in enumerate(starts.tolist()):
            mask[seq_i, window_i] = valid_base_window(arr[seq_i], int(start), int(start) + width)

    log_progress(f"window-ism: predicting {n_sequences} reference sequences", enabled=progress)
    reference = predictor.predict(arr).astype(np.float32, copy=False)
    if reference.shape != (n_sequences,):
        raise ValueError(f"predictor must return one scalar per sequence; got shape {reference.shape}")

    sums = np.zeros(mask.shape, dtype=np.float64)
    abs_sums = np.zeros(mask.shape, dtype=np.float64)
    square_sums = np.zeros(mask.shape, dtype=np.float64)
    counts = np.zeros(mask.shape, dtype=np.int32)
    mutant_batch: list[np.ndarray] = []
    mutant_meta: list[tuple[int, int]] = []

    def flush() -> None:
        """Predict queued mutants and update per-window moments."""

        if not mutant_batch:
            return
        predictions = predictor.predict(np.stack(mutant_batch, axis=0))
        if predictions.shape[0] != len(mutant_meta):
            raise ValueError("predictor returned an unexpected number of mutant predictions")
        for prediction, (seq_i, window_i) in zip(predictions, mutant_meta):
            delta = float(prediction - reference[seq_i])
            sums[seq_i, window_i] += delta
            abs_sums[seq_i, window_i] += abs(delta)
            square_sums[seq_i, window_i] += delta * delta
            counts[seq_i, window_i] += 1
        mutant_batch.clear()
        mutant_meta.clear()

    n_scored_windows = int(mask.sum())
    reporter = ProgressReporter(
        "window-ism: scan windows",
        total=n_scored_windows,
        unit="windows",
        enabled=progress,
    )
    for seq_i in range(n_sequences):
        for window_i in np.flatnonzero(mask[seq_i]).tolist():
            start = int(starts_out[seq_i, window_i])
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), seq_i, start]))
            for _ in range(int(n_ablations)):
                mutant = arr[seq_i].copy()
                scramble_window_inplace(
                    mutant,
                    start=start,
                    window_size=width,
                    strategy="random_different",
                    rng=rng,
                )
                mutant_batch.append(mutant)
                mutant_meta.append((seq_i, window_i))
                if len(mutant_batch) >= int(mutation_batch_size):
                    flush()
            reporter.update()
    flush()
    reporter.close(extra=f"{int(counts.sum())} mutants predicted")

    if n_scored_windows and not np.all(counts[mask] == int(n_ablations)):
        raise RuntimeError("Not all valid windows received the requested number of ablations")
    mean = np.zeros(mask.shape, dtype=np.float32)
    mean_abs = np.zeros(mask.shape, dtype=np.float32)
    std = np.zeros(mask.shape, dtype=np.float32)
    if n_scored_windows:
        mean_values = sums[mask] / counts[mask]
        mean_abs_values = abs_sums[mask] / counts[mask]
        variance = (square_sums[mask] / counts[mask]) - (mean_values * mean_values)
        variance = np.maximum(variance, 0.0)
        mean[mask] = mean_values.astype(np.float32)
        mean_abs[mask] = mean_abs_values.astype(np.float32)
        std[mask] = np.sqrt(variance).astype(np.float32)

    return WindowISMResult(
        window_starts=starts_out,
        window_mask=mask,
        mean_deltas=mean,
        mean_abs_deltas=mean_abs,
        std_deltas=std,
        reference_predictions=reference,
        valid_lengths=lengths,
        input_shape=tuple(int(value) for value in arr.shape),
        window_size=width,
        stride=step,
        n_ablations=int(n_ablations),
        seed=int(seed),
    )


def sequence_ids_sha256(sequence_ids: Sequence[str]) -> str:
    """Return a stable digest for an ordered collection of sequence IDs."""

    digest = hashlib.sha256()
    for sequence_id in sequence_ids:
        digest.update(str(sequence_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _coverage_counts(result: WindowISMResult) -> tuple[int, int]:
    """Return covered and uncovered valid-base counts for one result."""

    covered_total = 0
    valid_total = int(np.asarray(result.valid_lengths, dtype=np.int64).sum())
    for seq_i, valid_length in enumerate(result.valid_lengths.tolist()):
        covered = np.zeros(int(valid_length), dtype=bool)
        for window_i in np.flatnonzero(result.window_mask[seq_i]).tolist():
            start = int(result.window_starts[seq_i, window_i])
            covered[start : start + int(result.window_size)] = True
        covered_total += int(covered.sum())
    return covered_total, valid_total - covered_total


def save_window_ism_result(
    result: WindowISMResult,
    out_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
    sequence_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> None:
    """Save window-ISM arrays and reproducibility metadata."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"window-ism: saving results to {out}", enabled=progress)
    arrays = {
        "window_starts": result.window_starts,
        "window_mask": result.window_mask,
        "mean_deltas": result.mean_deltas,
        "mean_abs_deltas": result.mean_abs_deltas,
        "std_deltas": result.std_deltas,
        "reference_predictions": result.reference_predictions,
        "valid_lengths": result.valid_lengths,
    }
    for name, values in arrays.items():
        np.save(out / f"{name}.npy", np.asarray(values))

    candidate_mask = result.window_starts >= 0
    covered_bases, uncovered_bases = _coverage_counts(result)
    n_short = int(np.count_nonzero(result.valid_lengths < result.window_size))
    ids_digest = None
    if sequence_ids is not None:
        if len(sequence_ids) != int(result.valid_lengths.shape[0]):
            raise ValueError("sequence_ids length must match the number of result sequences")
        ids_digest = sequence_ids_sha256(sequence_ids)
    summary = {
        "analysis": "window_ism",
        "effect_definition": "mutant_prediction - reference_prediction",
        "mutation_policy": "random_different_every_base",
        "alternative_base_sampling": "uniform_over_other_three_bases",
        "window_size": int(result.window_size),
        "stride": int(result.stride),
        "n_ablations": int(result.n_ablations),
        "seed": int(result.seed),
        "coordinate_convention": "zero_based_half_open",
        "window_interval": "[start, start + window_size)",
        "terminal_window_anchored": True,
        "padding_start_value": -1,
        "masked_effect_fill_value": 0.0,
        "raw_replicates_saved": False,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset": str(dataset) if dataset is not None else None,
        "sequence_ids_sha256": ids_digest,
        "n_sequences": int(result.valid_lengths.shape[0]),
        "input_shape": list(result.input_shape),
        "window_effect_shape": list(result.mean_deltas.shape),
        "n_candidate_windows": int(candidate_mask.sum()),
        "n_scored_windows": int(result.window_mask.sum()),
        "n_ambiguous_windows": int(np.count_nonzero(candidate_mask & ~result.window_mask)),
        "n_sequences_shorter_than_window": n_short,
        "n_valid_bases": int(result.valid_lengths.sum()),
        "n_covered_valid_bases": covered_bases,
        "n_uncovered_valid_bases": uncovered_bases,
        "arrays": {
            name: {"shape": list(np.asarray(values).shape), "dtype": str(np.asarray(values).dtype)}
            for name, values in arrays.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("window-ism: done", enabled=progress)
