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
    fast: bool = True


# Each extra fastISM bucket may trigger a ~45-50s torch.export capture, while
# saving ~1M all-zero padding edits (~4 per position) recovers roughly 15-20s on
# GPU. Splitting only above ~2.5M wasted padding edits keeps small runs in a
# single capture while saving >85% of padding work on transcriptome-wide scans.
_FAST_ISM_WASTED_EDITS_PER_BUCKET = 2_500_000
_MAX_FAST_ISM_BUCKETS = 8
_MAX_SLOW_ISM_BUCKETS = 32


def _length_buckets(sorted_lengths: np.ndarray, *, fast: bool) -> list[slice]:
    """Partition sorted positive valid lengths into contiguous index slices."""

    m = int(sorted_lengths.size)
    if m <= 1:
        return [slice(0, m)]
    max_len = int(sorted_lengths[-1])
    wasted_edits = int(np.sum(4 * (max_len - sorted_lengths), dtype=np.int64))
    if wasted_edits <= 0:
        return [slice(0, m)]

    if fast:
        n_buckets = min(
            m,
            _MAX_FAST_ISM_BUCKETS,
            max(1, wasted_edits // _FAST_ISM_WASTED_EDITS_PER_BUCKET + 1),
        )
    else:
        n_buckets = min(m, _MAX_SLOW_ISM_BUCKETS)

    edges = np.linspace(0, m, n_buckets + 1, dtype=np.int64)
    buckets: list[slice] = []
    for start, end in zip(edges[:-1].tolist(), edges[1:].tolist()):
        if end <= start:
            continue
        if buckets and int(sorted_lengths[buckets[-1].stop - 1]) == int(sorted_lengths[end - 1]):
            buckets[-1] = slice(buckets[-1].start, end)
        else:
            buckets.append(slice(start, end))
    return buckets


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
    fast: bool = True,
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
        fast: Whether to enable Tangermeme verified fastISM when
            ``backend='tangermeme'``. Enabled by default; Tangermeme skips
            capture on short jobs, verifies accuracy on a sample batch, and
            falls back automatically when unsupported or slower.
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

    reference = np.zeros((N,), dtype=np.float32)
    zero_idx = np.flatnonzero(lengths == 0)
    if zero_idx.size > 0:
        reference[zero_idx] = predictor.predict(arr[zero_idx]).astype(np.float32, copy=False)

    active_idx = np.flatnonzero(lengths > 0)
    if active_idx.size > 0:
        # Probe differentiability once without copying X so non-Module callables
        # use fine-grained slow buckets directly.
        probe = prepare_tangermeme_input(
            predictor,
            arr[active_idx[:1]],
            batch_size=None,
            separate_annotations=True,
        )
        use_fast = bool(fast) and probe.differentiable
        eff_batch_size = (
            int(mutation_batch_size)
            if use_fast
            else min(int(mutation_batch_size), probe.batch_size)
        )
        active_lengths = lengths[active_idx]
        order = np.argsort(active_lengths, kind="stable")
        sorted_idx = active_idx[order]
        sorted_lengths = active_lengths[order]

        for sl in _length_buckets(sorted_lengths, fast=use_fast):
            sub_idx = sorted_idx[sl]
            bucket_max = int(sorted_lengths[sl.stop - 1])
            prepared = prepare_tangermeme_input(
                predictor,
                arr[sub_idx],
                batch_size=eff_batch_size,
                separate_annotations=True,
            )
            raw = saturation_mutagenesis(
                prepared.model,
                prepared.sequences,
                args=prepared.args,
                start=0,
                end=bucket_max,
                batch_size=prepared.batch_size,
                raw_outputs=True,
                dtype=torch.float32,
                device=prepared.device,
                verbose=progress,
                func=squeeze_prediction,
                fast=use_fast,
            )
            if not isinstance(raw.y0, torch.Tensor) or not isinstance(raw.y_hat, torch.Tensor):
                raise ValueError("predictor model must return one tensor")
            ref_bucket = raw.y0.numpy().astype(np.float32, copy=False).reshape(-1)
            values_bucket = raw.y_hat.numpy().astype(np.float32, copy=False)
            reference[sub_idx] = ref_bucket
            deltas[sub_idx, :, :bucket_max] = values_bucket - ref_bucket[:, None, None]

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
