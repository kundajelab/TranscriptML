from __future__ import annotations

import math
import re


CODONS_BY_AA = {
    "F": ("UUU", "UUC"),
    "L": ("UUA", "UUG", "CUU", "CUC", "CUA", "CUG"),
    "I": ("AUU", "AUC", "AUA"),
    "M": ("AUG",),
    "V": ("GUU", "GUC", "GUA", "GUG"),
    "S": ("UCU", "UCC", "UCA", "UCG", "AGU", "AGC"),
    "P": ("CCU", "CCC", "CCA", "CCG"),
    "T": ("ACU", "ACC", "ACA", "ACG"),
    "A": ("GCU", "GCC", "GCA", "GCG"),
    "Y": ("UAU", "UAC"),
    "H": ("CAU", "CAC"),
    "Q": ("CAA", "CAG"),
    "N": ("AAU", "AAC"),
    "K": ("AAA", "AAG"),
    "D": ("GAU", "GAC"),
    "E": ("GAA", "GAG"),
    "C": ("UGU", "UGC"),
    "W": ("UGG",),
    "R": ("CGU", "CGC", "CGA", "CGG", "AGA", "AGG"),
    "G": ("GGU", "GGC", "GGA", "GGG"),
    "Stop": ("UAA", "UAG", "UGA"),
}

AA_NAMES = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "Stop": "Stop",
}

AA_ORDER = tuple(CODONS_BY_AA.keys())
AA_SORT = {aa: i for i, aa in enumerate(AA_ORDER)}
CODON_ORDER = tuple(codon for codons in CODONS_BY_AA.values() for codon in codons)
CODON_SORT = {codon: i for i, codon in enumerate(CODON_ORDER)}
CODON_TO_AA = {codon: aa for aa, codons in CODONS_BY_AA.items() for codon in codons}


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename component."""

    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "unnamed"


def position_bin(codon_index: int, n_codons: int, n_bins: int) -> int:
    """Return the equal-width relative-position bin for a CDS codon index."""

    raw = int(math.floor((codon_index / n_codons) * n_bins))
    return min(max(raw, 0), n_bins - 1)


def bin_columns(bin_id: int, n_bins: int) -> dict[str, float | int]:
    """Return start/end/center columns for a relative-position bin."""

    return {
        "position_bin": bin_id,
        "bin_start": bin_id / n_bins,
        "bin_end": (bin_id + 1) / n_bins,
        "bin_center": (bin_id + 0.5) / n_bins,
    }
