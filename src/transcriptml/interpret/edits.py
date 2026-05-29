from __future__ import annotations

from typing import Sequence, Set

import numpy as np

from transcriptml.interpret.motifs import ALL_BASES, base_indices_from_ohe, region_matches_motif


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


def scramble_region_bases(orig: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    """Scramble a base-index vector using the requested strategy.

    Args:
        orig: Original base-index vector to scramble.
        strategy: Scrambling strategy, one of ``random_different``, ``shuffle``,
            or ``dinuc_shuffle``.
        rng: NumPy random generator used by the selected strategy.
    """

    if strategy == "random_different":
        return random_different_bases(orig, rng)
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
) -> None:
    """Scramble one fixed-width base window in place.

    Args:
        x: Encoded ``(C, L)`` sequence to modify in place.
        start: Zero-based inclusive window start.
        window_size: Number of bases to scramble.
        strategy: Scrambling strategy name supported by
            ``scramble_region_bases``.
        rng: NumPy random generator used for scrambling.
    """

    orig = extract_base_indices(x, start, start + window_size)
    write_bases_inplace(x, start, scramble_region_bases(orig, strategy, rng))


def scramble_motif_ablating_inplace(
    x: np.ndarray,
    *,
    motif_start: int,
    motif_sets: Sequence[Set[int]],
    strategy: str,
    rng: np.random.Generator,
    max_tries: int = 25,
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
    """

    start = int(motif_start)
    end = start + len(motif_sets)
    orig = extract_base_indices(x, start, end)
    for _ in range(max_tries):
        cand = scramble_region_bases(orig, strategy, rng)
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
