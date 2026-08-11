"""Spliced transcript coordinate models and interval conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Exon:
    """Half-open genomic exon with an assigned transcript interval."""

    chrom: str
    start: int
    end: int
    strand: str
    exon_number: str = ""
    exon_id: str = ""
    tx_start: int = 0
    tx_end: int = 0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid exon interval {self.chrom}:{self.start}-{self.end}")
        if self.strand not in {"+", "-"}:
            raise ValueError(f"invalid exon strand: {self.strand!r}")


@dataclass(frozen=True)
class Region:
    """Half-open transcript interval carrying a biological region label."""

    start: int
    end: int
    label: str


@dataclass
class Transcript:
    """One mature transcript in transcript 5-prime to 3-prime orientation."""

    transcript_id: str
    gene_id: str
    gene_name: str
    transcript_name: str
    transcript_type: str
    chrom: str
    strand: str
    exons: list[Exon] = field(default_factory=list)
    feature_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    offset: int = 0

    def finalize(self) -> None:
        """Validate exons, order them 5-prime to 3-prime, and assign coordinates."""

        if not self.exons:
            raise ValueError(f"transcript {self.transcript_id} has no exons")
        if any(e.chrom != self.chrom or e.strand != self.strand for e in self.exons):
            raise ValueError(f"inconsistent chromosome/strand in {self.transcript_id}")
        genomic = sorted(self.exons, key=lambda e: (e.start, e.end))
        for left, right in zip(genomic, genomic[1:]):
            if left.end > right.start:
                raise ValueError(f"overlapping exons in {self.transcript_id}")
        ordered = genomic if self.strand == "+" else list(reversed(genomic))
        cursor = 0
        for exon in ordered:
            exon.tx_start = cursor
            cursor += exon.end - exon.start
            exon.tx_end = cursor
        self.exons = ordered

    @property
    def length(self) -> int:
        return sum(e.end - e.start for e in self.exons)

    def genome_to_transcript(self, chrom: str, pos: int) -> int | None:
        """Map one zero-based genomic base to transcript space."""

        if chrom != self.chrom:
            return None
        for exon in self.exons:
            if exon.start <= pos < exon.end:
                delta = pos - exon.start if self.strand == "+" else exon.end - 1 - pos
                return exon.tx_start + delta
        return None

    def transcript_to_genome(self, pos: int) -> tuple[str, int, str]:
        """Map one zero-based transcript base to genomic coordinates."""

        if pos < 0 or pos >= self.length:
            raise IndexError(f"transcript position {pos} outside [0,{self.length})")
        for exon in self.exons:
            if exon.tx_start <= pos < exon.tx_end:
                delta = pos - exon.tx_start
                genomic = exon.start + delta if self.strand == "+" else exon.end - 1 - delta
                return self.chrom, genomic, self.strand
        raise AssertionError("finalized transcript has a coordinate gap")

    def genomic_interval_to_transcript(self, start: int, end: int) -> list[tuple[int, int]]:
        """Map a half-open genomic interval to covered transcript pieces."""

        pieces: list[tuple[int, int]] = []
        for exon in self.exons:
            lo, hi = max(start, exon.start), min(end, exon.end)
            if lo >= hi:
                continue
            if self.strand == "+":
                pieces.append((exon.tx_start + lo - exon.start, exon.tx_start + hi - exon.start))
            else:
                pieces.append((exon.tx_start + exon.end - hi, exon.tx_start + exon.end - lo))
        return sorted(pieces)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or directly adjacent half-open intervals."""

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def annotate_regions(tx: Transcript) -> list[Region]:
    """Partition a mature transcript into UTR/CDS or noncoding-exon sequence."""

    cds_genomic = tx.feature_intervals.get("CDS", []) + tx.feature_intervals.get("stop_codon", [])
    cds = merge_intervals(
        piece
        for interval in cds_genomic
        for piece in tx.genomic_interval_to_transcript(*interval)
    )
    if not cds:
        if tx.transcript_type == "protein_coding":
            raise ValueError(f"protein-coding transcript {tx.transcript_id} has no CDS annotation")
        return [Region(0, tx.length, "noncoding_exon")]
    cds_start = min(start for start, _ in cds)
    cds_end = max(end for _, end in cds)
    result: list[Region] = []
    if cds_start:
        result.append(Region(0, cds_start, "5putr"))
    result.append(Region(cds_start, cds_end, "cds"))
    if cds_end < tx.length:
        result.append(Region(cds_end, tx.length, "3putr"))
    return result
