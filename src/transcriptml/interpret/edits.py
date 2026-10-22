from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tangermeme.ersatz import dinucleotide_shuffle, randomize, shuffle

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
    end = int(start) + len(bases_arr)
    x[:4, start:end] = 0
    for j, base in enumerate(bases_arr):
        x[int(base), int(start) + j] = 1


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
    encoded = torch.nn.functional.one_hot(
        torch.as_tensor(bases), num_classes=4
    ).T.unsqueeze(0).to(torch.int8)
    seed = _tangermeme_seed(rng)
    if strategy == "random":
        edited = randomize(encoded, 0, bases.size, n=1, random_state=seed)
    elif strategy == "shuffle":
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
    encoded_array = np.asarray(x)[channels, :]
    if not np.all((encoded_array == 0) | (encoded_array == 1)):
        raise ValueError("Tangermeme nucleotide edits require hard binary base channels")
    encoded = torch.as_tensor(encoded_array).unsqueeze(0)
    seed = _tangermeme_seed(rng)
    if strategy == "random":
        edited = randomize(encoded, int(start), int(end), n=1, random_state=seed)
    elif strategy == "shuffle":
        edited = shuffle(encoded, int(start), int(end), n=1, random_state=seed)
    else:
        raise ValueError("strategy must be 'random' or 'shuffle'")
    x[channels, :] = edited[0, 0].cpu().numpy()


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
