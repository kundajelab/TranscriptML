from __future__ import annotations

from typing import Sequence, Set

import numpy as np

from transcriptml.interpret.motifs import ALL_BASES, base_indices_from_ohe, region_matches_motif


def extract_base_indices(x: np.ndarray, start: int, end: int) -> np.ndarray:
    bases = base_indices_from_ohe(np.asarray(x)[:4, start:end])
    if np.any(bases < 0):
        raise ValueError("Cannot edit a region containing all-zero or ambiguous base columns")
    return bases.astype(np.int64, copy=False)


def write_bases_inplace(x: np.ndarray, start: int, bases: Sequence[int]) -> None:
    bases_arr = np.asarray(bases, dtype=np.int64)
    end = int(start) + len(bases_arr)
    x[:4, start:end] = 0
    for j, base in enumerate(bases_arr):
        x[int(base), int(start) + j] = 1


def random_different_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    orig = np.asarray(orig, dtype=np.int64)
    r = rng.integers(0, 3, size=orig.shape[0], dtype=np.int64)
    return r + (r >= orig).astype(np.int64)


def shuffle_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(orig, dtype=np.int64)[rng.permutation(len(orig))].copy()


def dinuc_shuffle_bases(orig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small dinucleotide-preserving shuffle; falls back to base shuffle."""

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
    try:
        extract_base_indices(x, start, end)
    except ValueError:
        return False
    return True
