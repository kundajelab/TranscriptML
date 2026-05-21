from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np

BASE_TO_INDEX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "U"}
ALL_BASES: Set[int] = {0, 1, 2, 3}


def parse_motif(motif: str) -> list[set[int]]:
    """Parse motifs with A/C/G/U/T, bracket alternatives, and N/./X wildcards."""

    s = motif.strip().upper()
    out: list[set[int]] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "[":
            j = s.find("]", i + 1)
            if j == -1:
                raise ValueError(f"Unmatched '[' in motif: {motif}")
            parts = [p.strip().upper() for p in s[i + 1 : j].split("|") if p.strip()]
            if not parts:
                raise ValueError(f"Empty bracket group in motif: {motif}")
            allowed: set[int] = set()
            for p in parts:
                if len(p) != 1:
                    raise ValueError(f"Bracket alternatives must be single bases in motif: {motif}")
                if p in {"N", ".", "X"}:
                    allowed |= ALL_BASES
                elif p in BASE_TO_INDEX:
                    allowed.add(BASE_TO_INDEX[p])
                else:
                    raise ValueError(f"Unknown base '{p}' in motif: {motif}")
            out.append(allowed)
            i = j + 1
        elif ch == "]":
            raise ValueError(f"Unmatched ']' in motif: {motif}")
        else:
            if ch in {"N", ".", "X"}:
                out.append(set(ALL_BASES))
            elif ch in BASE_TO_INDEX:
                out.append({BASE_TO_INDEX[ch]})
            else:
                raise ValueError(f"Unknown base '{ch}' in motif: {motif}")
            i += 1
    if not out:
        raise ValueError("Motif parsed to length 0")
    return out


def motif_length(motif: str | Sequence[Set[int]]) -> int:
    return len(parse_motif(motif)) if isinstance(motif, str) else len(motif)


def base_indices_from_ohe(ohe_4_by_L: np.ndarray) -> np.ndarray:
    """Return base index per position, or -1 for all-zero/ambiguous columns."""

    arr = np.asarray(ohe_4_by_L)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError(f"Expected at least four base channels, got {arr.shape}")
    bases = np.full(arr.shape[1], -1, dtype=np.int64)
    base = arr[:4]
    counts = np.count_nonzero(base, axis=0)
    valid = counts == 1
    bases[valid] = np.argmax(base[:, valid], axis=0).astype(np.int64)
    return bases


def region_matches_motif(base_region: np.ndarray, motif_sets: Sequence[Set[int]]) -> bool:
    if len(base_region) != len(motif_sets):
        return False
    for base, allowed in zip(base_region, motif_sets):
        if int(base) < 0 or int(base) not in allowed:
            return False
    return True


def find_motif_starts(ohe_4_by_L: np.ndarray, motif: str | Sequence[Set[int]]) -> np.ndarray:
    motif_sets = parse_motif(motif) if isinstance(motif, str) else [set(x) for x in motif]
    m = len(motif_sets)
    bases = base_indices_from_ohe(ohe_4_by_L)
    L = len(bases)
    if m > L:
        return np.empty((0,), dtype=np.int64)
    valid = np.ones(L - m + 1, dtype=bool)
    for offset, allowed in enumerate(motif_sets):
        allowed_arr = np.fromiter((int(x) for x in allowed), dtype=np.int64)
        valid &= np.isin(bases[offset : offset + valid.size], allowed_arr)
    return np.nonzero(valid)[0].astype(np.int64, copy=False)


def intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return int(a0) < int(b1) and int(b0) < int(a1)
