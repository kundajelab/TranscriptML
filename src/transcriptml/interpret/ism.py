from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.interpret.backends import normalize_backend, prepare_tangermeme_input
from transcriptml.interpret.predictor import Predictor
from transcriptml.models.common import squeeze_prediction
from transcriptml.progress import ProgressReporter, log_progress


@dataclass
class ISMResult:
    deltas: np.ndarray
    reference_predictions: np.ndarray
    valid_lengths: np.ndarray
    backend: str = "tangermeme"
    fast: bool = False


def _normalize_lengths(arr: np.ndarray, valid_lengths: np.ndarray | None) -> np.ndarray:
    """Infer and validate sequence lengths for an ISM batch."""

    lengths = infer_valid_lengths(arr) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int64)
    if lengths.shape != (arr.shape[0],):
        raise ValueError(f"valid_lengths must have shape ({arr.shape[0]},), got {lengths.shape}")
    if np.any(lengths < 0) or np.any(lengths > arr.shape[-1]):
        raise ValueError(f"valid_lengths entries must be between 0 and encoded length {arr.shape[-1]}")
    return lengths.astype(np.int64, copy=False)


def _compute_transcriptml_ism(
    arr: np.ndarray,
    predictor: Predictor,
    *,
    lengths: np.ndarray,
    mutation_batch_size: int,
    progress: bool,
) -> ISMResult:
    """Historical TranscriptML single-substitution implementation."""

    N, _, L = arr.shape
    log_progress(f"ism: predicting {N} reference sequences", enabled=progress)
    ref = predictor.predict(arr)
    deltas = np.zeros((N, 4, L), dtype=np.float32)
    batch: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    n_mutants = 0

    def flush() -> None:
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
    return ISMResult(
        deltas=deltas,
        reference_predictions=ref.astype(np.float32),
        valid_lengths=lengths,
        backend="transcriptml",
        fast=False,
    )


def compute_ism(
    X: np.ndarray,
    predictor: Predictor,
    *,
    valid_lengths: np.ndarray | None = None,
    mutation_batch_size: int = 512,
    backend: str = "tangermeme",
    fast: bool = False,
    progress: bool = True,
) -> ISMResult:
    """Single-nucleotide ISM with signed mutant-minus-reference effects.

    The returned ``deltas`` array has shape ``(N, 4, L)``. At each valid base
    position, the three alternative base channels store mutant-reference
    deltas; the original-base channel remains zero.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch with at least four base
            channels.
        predictor: Predictor used to score reference and mutant sequences.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred from ``X``.
        mutation_batch_size: Maximum number of mutant sequences to score in one
            prediction batch.
        backend: Interpretation backend. ``tangermeme`` is the default;
            ``transcriptml`` selects the historical enumerator.
        fast: Opt in to Tangermeme fastISM. Tangermeme verifies the fast path
            and falls back when unsupported. Disabled by default because
            capture provides little benefit for globally coupled models such
            as Saluki and LegNet.
        progress: Whether to emit progress messages while scanning.
    """

    arr = np.asarray(X)
    if arr.ndim != 3 or arr.shape[1] < 4:
        raise ValueError(f"Expected X with shape (N, C>=4, L), got {arr.shape}")
    if int(mutation_batch_size) <= 0:
        raise ValueError("mutation_batch_size must be positive")
    N, _, L = arr.shape
    lengths = _normalize_lengths(arr, valid_lengths)
    selected_backend = normalize_backend(backend)
    if selected_backend == "transcriptml":
        if fast:
            raise ValueError("fast=True is available only with backend='tangermeme'")
        return _compute_transcriptml_ism(
            arr,
            predictor,
            lengths=lengths,
            mutation_batch_size=int(mutation_batch_size),
            progress=progress,
        )

    bases = arr[:, :4]
    if not np.all((bases == 0) | (bases == 1)):
        raise ValueError(
            "Tangermeme ISM requires hard binary nucleotide channels; "
            "use backend='transcriptml' for soft encodings"
        )
    deltas = np.zeros((N, 4, L), dtype=np.float32)
    if N == 0:
        return ISMResult(
            deltas=deltas,
            reference_predictions=np.empty((0,), dtype=np.float32),
            valid_lengths=lengths,
            backend=selected_backend,
            fast=bool(fast),
        )

    max_length = int(lengths.max(initial=0))
    if max_length == 0:
        reference = predictor.predict(arr).astype(np.float32, copy=False)
    else:
        prepared = prepare_tangermeme_input(
            predictor,
            arr,
            batch_size=int(mutation_batch_size),
            separate_annotations=True,
        )
        raw = saturation_mutagenesis(
            prepared.model,
            prepared.sequences,
            args=prepared.args,
            start=0,
            end=max_length,
            batch_size=prepared.batch_size,
            raw_outputs=True,
            dtype=torch.float32,
            device=prepared.device,
            verbose=progress,
            func=squeeze_prediction,
            fast=bool(fast),
        )
        if not isinstance(raw.y0, torch.Tensor) or not isinstance(raw.y_hat, torch.Tensor):
            raise ValueError("predictor model must return one tensor")
        reference = raw.y0.numpy().astype(np.float32, copy=False).reshape(-1)
        values = raw.y_hat.numpy().astype(np.float32, copy=False)
        deltas[:, :, :max_length] = values - reference[:, None, None]

    positions = np.arange(L, dtype=np.int64)[None, :]
    valid = positions < lengths[:, None]
    unambiguous = np.count_nonzero(bases, axis=1) == 1
    deltas *= (valid & unambiguous)[:, None, :]
    return ISMResult(
        deltas=deltas,
        reference_predictions=reference,
        valid_lengths=lengths,
        backend=selected_backend,
        fast=bool(fast),
    )


def max_abs_effect_per_position(deltas: np.ndarray) -> np.ndarray:
    """Summarize ISM effects by maximum absolute base substitution effect.

    Args:
        deltas: ISM delta array with shape ``(N, 4, L)``.
    """

    return np.max(np.abs(np.asarray(deltas)), axis=1)


def save_ism_result(result: ISMResult, out_dir: str | Path, *, progress: bool = True) -> None:
    """Save ISM result arrays and a small JSON summary.

    Args:
        result: ISM result object to serialize.
        out_dir: Destination directory for arrays and summary JSON.
        progress: Whether to emit progress messages while saving.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"ism: saving results to {out}", enabled=progress)
    np.save(out / "deltas.npy", result.deltas)
    np.save(out / "reference_predictions.npy", result.reference_predictions)
    np.save(out / "valid_lengths.npy", result.valid_lengths)
    np.save(out / "max_abs_effect.npy", max_abs_effect_per_position(result.deltas))
    summary = {
        "analysis": "single_nucleotide_ism",
        "backend": result.backend,
        "fast_ism_requested": bool(result.fast),
        "delta_definition": "mutant_prediction - reference_prediction",
        "deltas_shape": list(result.deltas.shape),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("ism: done", enabled=progress)
