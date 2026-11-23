from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tangermeme.ersatz import dinucleotide_shuffle, shuffle
from tangermeme.utils import random_one_hot

from transcriptml.interpret.backends import normalize_backend
from transcriptml.interpret.motifs import (
    ALL_BASES,
    base_indices_from_ohe,
    region_matches_motif,
)


def effective_perturbation_backend(strategy: str, backend: str) -> str:
    """Return the implementation actually used for a scramble strategy."""

    selected_backend = normalize_backend(backend)
    return "transcriptml" if strategy == "random_different" else selected_backend


def extract_base_indices(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract unambiguous base indices from a sequence window.

    Args:
        x: Encoded ``(C, L)`` sequence with base channels first.
        start: Zero-based inclusive window start.
        end: Zero-based exclusive window end.
    """

    bases = base_indices_from_ohe(np.asarray(x)[:4, start:end])
    if np.any(bases < 0):
        raise ValueError("Cannot edit a region containing all-zero or ambiguous base columns")
    return bases.astype(np.int64, copy=False)


def write_bases_inplace(x: np.ndarray, start: int, bases: Sequence[int]) -> None:
    """Overwrite base channels in place while preserving annotation channels.

    Args:
        x: Encoded ``(C, L)`` sequence to modify in place.
        start: Zero-based position where replacement bases begin.
        bases: Base indices to write, using A/C/G/U offsets ``0`` through ``3``.
    """

    bases_arr = np.asarray(bases, dtype=np.int64)
    s = int(start)
    end = s + len(bases_arr)
    x[:4, s:end] = 0
    if bases_arr.size:
        x[bases_arr, s + np.arange(bases_arr.size, dtype=np.int64)] = 1


def random_different_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample bases that differ at every position from the original bases.

    Args:
        orig: Original base-index vector using A/C/G/U offsets ``0`` through
            ``3``.
        rng: NumPy random generator used for sampling.
    """

    orig = np.asarray(orig, dtype=np.int64)
    r = rng.integers(0, 3, size=orig.shape[0], dtype=np.int64)
    return r + (r >= orig).astype(np.int64)


def shuffle_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation of base indices.

    Args:
        orig: Original base-index vector to permute.
        rng: NumPy random generator used for the permutation.
    """

    return np.asarray(orig, dtype=np.int64)[rng.permutation(len(orig))].copy()


def dinuc_shuffle_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small dinucleotide-preserving shuffle; falls back to base shuffle.

    Args:
        orig: Original base-index vector to shuffle while preserving
            dinucleotide transitions when possible.
        rng: NumPy random generator used for edge ordering and fallbacks.
    """

    orig = np.asarray(orig, dtype=np.int64)
    n = int(orig.shape[0])
    if n < 2:
        return orig.copy()
    start, end = int(orig[0]), int(orig[-1])
    adj: list[list[int]] = [[] for _ in range(4)]
    indeg = np.zeros(4, dtype=np.int64)
    outdeg = np.zeros(4, dtype=np.int64)
    for i in range(n - 1):
        u, v = int(orig[i]), int(orig[i + 1])
        adj[u].append(v)
        outdeg[u] += 1
        indeg[v] += 1
    for edges in adj:
        rng.shuffle(edges)
    if start == end:
        if not np.all(indeg == outdeg):
            return shuffle_bases(orig, rng)
    else:
        for v in range(4):
            if v == start and outdeg[v] != indeg[v] + 1:
                return shuffle_bases(orig, rng)
            if v == end and indeg[v] != outdeg[v] + 1:
                return shuffle_bases(orig, rng)
            if v not in {start, end} and indeg[v] != outdeg[v]:
                return shuffle_bases(orig, rng)
    stack = [start]
    path: list[int] = []
    while stack:
        v = stack[-1]
        if adj[v]:
            stack.append(adj[v].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    if len(path) != n:
        return shuffle_bases(orig, rng)
    return np.array(path, dtype=np.int64)


def _tangermeme_seed(rng: np.random.Generator) -> int:
    """Draw a deterministic legacy-RandomState-compatible seed."""

    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _tangermeme_scramble(
    orig: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one Tangermeme ersatz edit to a base-index vector."""

    bases = np.asarray(orig, dtype=np.int64)
    if bases.size == 0:
        return bases.copy()
    seed = _tangermeme_seed(rng)
    if strategy == "random":
        edited_ohe = random_one_hot((1, 4, int(bases.size)), dtype="int8", random_state=seed)
        return edited_ohe[0].argmax(dim=0).cpu().numpy().astype(np.int64, copy=False)
    encoded = torch.nn.functional.one_hot(
        torch.as_tensor(bases), num_classes=4
    ).T.unsqueeze(0).to(torch.int8)
    if strategy == "shuffle":
        edited = shuffle(encoded, 0, bases.size, n=1, random_state=seed)
    elif strategy == "dinuc_shuffle":
        if bases.size < 2:
            return bases.copy()
        edited = dinucleotide_shuffle(
            encoded,
            0,
            bases.size,
            n=1,
            random_state=seed,
        )
    else:  # pragma: no cover - caller validates the fixed strategy set.
        raise ValueError(f"Unknown Tangermeme scramble strategy '{strategy}'")
    return edited[0, 0].argmax(dim=0).cpu().numpy().astype(np.int64, copy=False)


def tangermeme_region_edits_batch(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: Sequence[int],
    strategy: str,
    n_edits: int,
    seed: int,
    validate_binary: bool = True,
) -> np.ndarray:
    """Generate ``n_edits`` Tangermeme region perturbations while preserving annotations."""

    channels = np.asarray(base_channels, dtype=np.int64)
    if channels.shape != (4,):
        raise ValueError("Tangermeme nucleotide edits require exactly four base channels")
    if strategy not in {"random", "shuffle"}:
        raise ValueError("strategy must be 'random' or 'shuffle'")

    arr = np.asarray(x)
    if validate_binary:
        base_view = arr[channels, :]
        if not np.all((base_view == 0) | (base_view == 1)):
            raise ValueError("Tangermeme nucleotide edits require hard binary base channels")

    count = int(n_edits)
    mutants = np.repeat(arr[None], count, axis=0)
    s, e = int(start), int(end)
    reg_len = e - s
    if count <= 0 or reg_len <= 0:
        return mutants

    if strategy == "random":
        edited = (
            random_one_hot((count, 4, reg_len), dtype="int8", random_state=int(seed))
            .cpu()
            .numpy()
            .astype(arr.dtype, copy=False)
        )
    else:
        reg = np.ascontiguousarray(arr[channels, s:e], dtype=np.int8)
        edited = (
            shuffle(torch.as_tensor(reg).unsqueeze(0), 0, reg_len, n=count, random_state=int(seed))[0]
            .cpu()
            .numpy()
            .astype(arr.dtype, copy=False)
        )

    if np.array_equal(channels, np.arange(4, dtype=np.int64)):
        mutants[:, :4, s:e] = edited
    else:
        mutants[:, channels, s:e] = edited
    return mutants


def tangermeme_region_edit_inplace(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: Sequence[int],
    strategy: str,
    rng: np.random.Generator,
) -> None:
    """Apply a Tangermeme ersatz edit while preserving non-base channels.

    ``base_channels`` may select nucleotide channels anywhere in the encoded
    tensor; their ordering is preserved because ersatz operations treat the
    alphabet symbols generically.
    """

    channels = np.asarray(base_channels, dtype=np.int64)
    if channels.shape != (4,):
        raise ValueError("Tangermeme nucleotide edits require exactly four base channels")
    if strategy not in {"random", "shuffle"}:
        raise ValueError("strategy must be 'random' or 'shuffle'")
    encoded_array = np.asarray(x)[channels, :]
    if not np.all((encoded_array == 0) | (encoded_array == 1)):
        raise ValueError("Tangermeme nucleotide edits require hard binary base channels")
    s, e = int(start), int(end)
    reg_len = e - s
    if reg_len <= 0:
        return
    seed = _tangermeme_seed(rng)
    if strategy == "random":
        edited = random_one_hot((1, 4, reg_len), dtype="int8", random_state=seed)[0].cpu().numpy()
    else:
        reg = torch.as_tensor(np.ascontiguousarray(encoded_array[:, s:e], dtype=np.int8)).unsqueeze(0)
        edited = shuffle(reg, 0, reg_len, n=1, random_state=seed)[0, 0].cpu().numpy()
    x[channels, s:e] = edited


def scramble_region_bases(
    orig: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    *,
    backend: str = "tangermeme",
) -> np.ndarray:
    """Scramble a base-index vector using the requested strategy.

    Args:
        orig: Original base-index vector to scramble.
        strategy: Scrambling strategy, one of ``random``,
            ``random_different``, ``shuffle``, or ``dinuc_shuffle``.
        rng: NumPy random generator used by the selected strategy.
        backend: Perturbation backend. Tangermeme implements ``random``,
            ``shuffle``, and ``dinuc_shuffle``. The TranscriptML-only
            ``random_different`` policy is identical for both backend values.
    """

    selected_backend = normalize_backend(backend)
    if strategy == "random_different":
        return random_different_bases(orig, rng)
    if selected_backend == "tangermeme" and strategy in {"random", "shuffle", "dinuc_shuffle"}:
        return _tangermeme_scramble(orig, strategy, rng)
    if strategy == "random":
        return rng.integers(0, 4, size=np.asarray(orig).shape[0], dtype=np.int64)
    if strategy == "shuffle":
        return shuffle_bases(orig, rng)
    if strategy == "dinuc_shuffle":
        return dinuc_shuffle_bases(orig, rng)
    raise ValueError(f"Unknown scramble strategy '{strategy}'")


def scramble_window_inplace(
    x: np.ndarray,
    *,
    start: int,
    window_size: int,
    strategy: str,
    rng: np.random.Generator,
    backend: str = "tangermeme",
) -> None:
    """Scramble one fixed-width base window in place.

    Args:
        x: Encoded ``(C, L)`` sequence to modify in place.
        start: Zero-based inclusive window start.
        window_size: Number of bases to scramble.
        strategy: Scrambling strategy name supported by
            ``scramble_region_bases``.
        rng: NumPy random generator used for scrambling.
        backend: Perturbation backend.
    """

    orig = extract_base_indices(x, start, start + window_size)
    write_bases_inplace(
        x,
        start,
        scramble_region_bases(orig, strategy, rng, backend=backend),
    )


def scramble_motif_ablating_inplace(
    x: np.ndarray,
    *,
    motif_start: int,
    motif_sets: Sequence[set[int]],
    strategy: str,
    rng: np.random.Generator,
    max_tries: int = 25,
    backend: str = "tangermeme",
) -> None:
    """Scramble a motif instance until it no longer matches the motif.

    Args:
        x: Encoded ``(C, L)`` sequence to modify in place.
        motif_start: Zero-based motif start position.
        motif_sets: Parsed motif position sets describing allowed bases at each
            motif position.
        strategy: Scrambling strategy name supported by
            ``scramble_region_bases``.
        rng: NumPy random generator used for scrambling.
        max_tries: Maximum attempts before falling back to guaranteed
            motif-disrupting substitutions.
        backend: Perturbation backend.
    """

    start = int(motif_start)
    end = start + len(motif_sets)
    orig = extract_base_indices(x, start, end)
    for _ in range(max_tries):
        cand = scramble_region_bases(orig, strategy, rng, backend=backend)
        if not np.array_equal(cand, orig) and not region_matches_motif(cand, motif_sets):
            write_bases_inplace(x, start, cand)
            return
    for _ in range(max_tries):
        cand = random_different_bases(orig, rng)
        if not region_matches_motif(cand, motif_sets):
            write_bases_inplace(x, start, cand)
            return
    cand = random_different_bases(orig, rng)
    for j, allowed in enumerate(motif_sets):
        disallowed = list(ALL_BASES - set(allowed))
        if disallowed:
            cand[j] = int(rng.choice(disallowed))
            write_bases_inplace(x, start, cand)
            break
    else:
        raise ValueError("Motif cannot be ablated because every position allows every base")


def _draw_scramble_candidates_batch(
    orig: np.ndarray,
    *,
    n_draw: int,
    strategy: str,
    rng: np.random.Generator,
    backend: str,
) -> np.ndarray:
    """Draw ``(n_draw, len(orig))`` base-index candidates in one vectorized batch."""

    bases = np.asarray(orig, dtype=np.int64)
    k = int(bases.size)
    count = int(n_draw)
    if count <= 0 or k == 0:
        return np.empty((max(count, 0), k), dtype=np.int64)

    selected_backend = normalize_backend(backend)
    if strategy == "random_different":
        r = rng.integers(0, 3, size=(count, k), dtype=np.int64)
        return r + (r >= bases[None, :]).astype(np.int64)

    if selected_backend == "tangermeme" and strategy in {"random", "shuffle", "dinuc_shuffle"}:
        if strategy == "random":
            seed = _tangermeme_seed(rng)
            ohe = random_one_hot((count, 4, k), dtype="int8", random_state=seed)
            return ohe.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
        enc = torch.nn.functional.one_hot(
            torch.as_tensor(bases), num_classes=4
        ).T.unsqueeze(0).to(torch.int8)
        if strategy == "shuffle":
            seed = _tangermeme_seed(rng)
            edited = shuffle(enc, 0, k, n=count, random_state=seed)[0]
            return edited.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
        if k < 2:
            return np.repeat(bases[None, :], count, axis=0)
        # Expand across batch axis with n=1 and per-row seeds so Tangermeme's
        # Numba kernel runs once without raising ValueError on conserved paths.
        seeds = rng.integers(0, 2**31 - 1, size=count, dtype=np.int32)
        edited = dinucleotide_shuffle(enc.expand(count, -1, -1), 0, k, n=1, random_state=seeds)[:, 0]
        return edited.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)

    if strategy == "random":
        return rng.integers(0, 4, size=(count, k), dtype=np.int64)
    if strategy == "shuffle":
        return np.stack([shuffle_bases(bases, rng) for _ in range(count)], axis=0)
    if strategy == "dinuc_shuffle":
        return np.stack([dinuc_shuffle_bases(bases, rng) for _ in range(count)], axis=0)
    raise ValueError(f"Unknown scramble strategy '{strategy}'")


def scramble_window_batch_ohe(
    orig: np.ndarray,
    *,
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
    backend: str = "tangermeme",
    dtype: np.dtype | type = np.uint8,
) -> np.ndarray:
    """Return ``(n_scrambles, 4, len(orig))`` one-hot window scrambles."""

    bases = np.asarray(orig, dtype=np.int64)
    k = int(bases.size)
    count = int(n_scrambles)
    if count <= 0 or k == 0:
        return np.zeros((max(count, 0), 4, k), dtype=dtype)

    selected_backend = normalize_backend(backend)
    if selected_backend == "tangermeme" and strategy == "random":
        seed = _tangermeme_seed(rng)
        return (
            random_one_hot((count, 4, k), dtype="int8", random_state=seed)
            .cpu()
            .numpy()
            .astype(dtype, copy=False)
        )
    if selected_backend == "tangermeme" and strategy == "shuffle":
        seed = _tangermeme_seed(rng)
        enc = torch.nn.functional.one_hot(
            torch.as_tensor(bases), num_classes=4
        ).T.unsqueeze(0).to(torch.int8)
        return shuffle(enc, 0, k, n=count, random_state=seed)[0].cpu().numpy().astype(dtype, copy=False)
    if selected_backend == "tangermeme" and strategy == "dinuc_shuffle" and k >= 2:
        seeds = rng.integers(0, 2**31 - 1, size=count, dtype=np.int32)
        enc = torch.nn.functional.one_hot(
            torch.as_tensor(bases), num_classes=4
        ).T.unsqueeze(0).to(torch.int8)
        return (
            dinucleotide_shuffle(enc.expand(count, -1, -1), 0, k, n=1, random_state=seeds)[:, 0]
            .cpu()
            .numpy()
            .astype(dtype, copy=False)
        )

    cands = _draw_scramble_candidates_batch(
        bases, n_draw=count, strategy=strategy, rng=rng, backend=selected_backend
    )
    return (cands[:, None, :] == np.arange(4, dtype=np.int64)[None, :, None]).astype(
        dtype, copy=False
    )


def scramble_motif_ablating_batch(
    orig: np.ndarray,
    *,
    motif_sets: Sequence[set[int]],
    n_scrambles: int,
    strategy: str,
    rng: np.random.Generator,
    max_tries: int = 25,
    backend: str = "tangermeme",
) -> np.ndarray:
    """Generate ``(n_scrambles, len(motif_sets))`` motif-disrupting base-index rows."""

    if all(len(allowed) >= 4 for allowed in motif_sets):
        raise ValueError("Motif cannot be ablated because every position allows every base")

    bases = np.asarray(orig, dtype=np.int64)
    k = len(motif_sets)
    count = int(n_scrambles)
    if count <= 0:
        return np.empty((0, k), dtype=np.int64)

    allowed_mask = np.zeros((k, 4), dtype=bool)
    for j, allowed in enumerate(motif_sets):
        for base in allowed:
            if 0 <= int(base) < 4:
                allowed_mask[j, int(base)] = True

    def valid_rows(cands: np.ndarray) -> np.ndarray:
        differs = np.any(cands != bases[None, :], axis=1)
        matches = np.all(allowed_mask[np.arange(k)[None, :], cands], axis=1)
        return cands[differs & ~matches]

    out = np.empty((count, k), dtype=np.int64)
    filled = 0
    total_budget = max(count * int(max_tries), count)
    drawn = 0
    while filled < count and drawn < total_budget:
        step = min(total_budget - drawn, max((count - filled) * 4, 32))
        cands = _draw_scramble_candidates_batch(
            bases, n_draw=step, strategy=strategy, rng=rng, backend=backend
        )
        drawn += step
        accepted = valid_rows(cands)
        if accepted.shape[0] > 0:
            take = min(count - filled, int(accepted.shape[0]))
            out[filled : filled + take] = accepted[:take]
            filled += take
        elif strategy == "dinuc_shuffle" and np.all(cands == bases[None, :]):
            # Unique Eulerian path (or homopolymer): dinuc_shuffle can never alter
            # orig, so skip remaining doomed retries and fall through immediately.
            break

    if filled < count:
        need = count - filled
        fallback_cands = _draw_scramble_candidates_batch(
            bases,
            n_draw=max(need * int(max_tries), 32),
            strategy="random_different",
            rng=rng,
            backend="transcriptml",
        )
        accepted = valid_rows(fallback_cands)
        if accepted.shape[0] > 0:
            take = min(need, int(accepted.shape[0]))
            out[filled : filled + take] = accepted[:take]
            filled += take

    if filled < count:
        disallowed_by_pos = [
            (j, sorted(ALL_BASES - set(allowed)))
            for j, allowed in enumerate(motif_sets)
            if ALL_BASES - set(allowed)
        ]
        if not disallowed_by_pos:
            raise ValueError("Motif cannot be ablated because every position allows every base")
        pos_j, disallowed = disallowed_by_pos[0]
        while filled < count:
            cand = random_different_bases(bases, rng)
            cand[pos_j] = int(rng.choice(disallowed))
            out[filled] = cand
            filled += 1

    return out


def valid_base_window(x: np.ndarray, start: int, end: int) -> bool:
    """Return whether a window contains only unambiguous base columns.

    Args:
        x: Encoded ``(C, L)`` sequence with base channels first.
        start: Zero-based inclusive window start.
        end: Zero-based exclusive window end.
    """

    try:
        extract_base_indices(x, start, end)
    except ValueError:
        return False
    return True
