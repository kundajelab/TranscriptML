"""Transcript-oriented mature-transcript and full-gene coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


COORDINATE_SPACES = ("mature_transcript", "gene")


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
    """One selected transcript/gene in RNA 5-prime to 3-prime orientation."""

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
    coordinate_space: str = "mature_transcript"
    genomic_start: int | None = None
    genomic_end: int | None = None

    def finalize(self) -> None:
        """Validate exons, order them 5-prime to 3-prime, and assign coordinates."""

        if not self.exons:
            raise ValueError(f"transcript {self.transcript_id} has no exons")
        if any(e.chrom != self.chrom or e.strand != self.strand for e in self.exons):
            raise ValueError(f"inconsistent chromosome/strand in {self.transcript_id}")
        if self.coordinate_space not in COORDINATE_SPACES:
            raise ValueError(f"invalid coordinate space: {self.coordinate_space!r}")
        genomic = sorted(self.exons, key=lambda e: (e.start, e.end))
        for left, right in zip(genomic, genomic[1:]):
            if left.end > right.start:
                raise ValueError(f"overlapping exons in {self.transcript_id}")
        exon_start = genomic[0].start
        exon_end = genomic[-1].end
        if self.genomic_start is None:
            self.genomic_start = exon_start
        if self.genomic_end is None:
            self.genomic_end = exon_end
        # The selected gene/transcript span is intentionally bounded by its
        # first and last exon. GTF transcript rows should agree, but using the
        # actual annotated sequence prevents terminal non-exonic padding.
        if self.genomic_start > exon_start or self.genomic_end < exon_end:
            raise ValueError(f"transcript span does not contain all exons in {self.transcript_id}")
        self.genomic_start = exon_start
        self.genomic_end = exon_end
        ordered = genomic if self.strand == "+" else list(reversed(genomic))
        if self.coordinate_space == "mature_transcript":
            cursor = 0
            for exon in ordered:
                exon.tx_start = cursor
                cursor += exon.end - exon.start
                exon.tx_end = cursor
        else:
            for exon in ordered:
                if self.strand == "+":
                    exon.tx_start = exon.start - self.genomic_start
                    exon.tx_end = exon.end - self.genomic_start
                else:
                    exon.tx_start = self.genomic_end - exon.end
                    exon.tx_end = self.genomic_end - exon.start
        self.exons = ordered

    @property
    def length(self) -> int:
        if self.coordinate_space == "gene":
            if self.genomic_start is None or self.genomic_end is None:
                raise ValueError(f"transcript {self.transcript_id} has not been finalized")
            return self.genomic_end - self.genomic_start
        return sum(e.end - e.start for e in self.exons)

    def genome_to_transcript(self, chrom: str, pos: int) -> int | None:
        """Map one zero-based genomic base to transcript space."""

        if chrom != self.chrom:
            return None
        if self.coordinate_space == "gene":
            assert self.genomic_start is not None and self.genomic_end is not None
            if self.genomic_start <= pos < self.genomic_end:
                return pos - self.genomic_start if self.strand == "+" else self.genomic_end - 1 - pos
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
        if self.coordinate_space == "gene":
            assert self.genomic_start is not None and self.genomic_end is not None
            genomic = self.genomic_start + pos if self.strand == "+" else self.genomic_end - 1 - pos
            return self.chrom, genomic, self.strand
        for exon in self.exons:
            if exon.tx_start <= pos < exon.tx_end:
                delta = pos - exon.tx_start
                genomic = exon.start + delta if self.strand == "+" else exon.end - 1 - delta
                return self.chrom, genomic, self.strand
        raise AssertionError("finalized transcript has a coordinate gap")

    def genomic_interval_to_transcript(self, start: int, end: int) -> list[tuple[int, int]]:
        """Map a half-open genomic interval to covered transcript pieces."""

        if self.coordinate_space == "gene":
            assert self.genomic_start is not None and self.genomic_end is not None
            lo, hi = max(start, self.genomic_start), min(end, self.genomic_end)
            if lo >= hi:
                return []
            if self.strand == "+":
                return [(lo - self.genomic_start, hi - self.genomic_start)]
            return [(self.genomic_end - hi, self.genomic_end - lo)]
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


def _gene_regions(tx: Transcript, cds: list[tuple[int, int]]) -> list[Region]:
    """Partition a full-gene locus into exon-derived labels and introns."""

    coding = bool(cds)
    if not coding and tx.transcript_type == "protein_coding":
        raise ValueError(f"protein-coding transcript {tx.transcript_id} has no CDS annotation")
    cds_start = min((start for start, _ in cds), default=0)
    cds_end = max((end for _, end in cds), default=0)
    labeled_exons: list[Region] = []
    for exon in sorted(tx.exons, key=lambda item: item.tx_start):
        boundaries = [exon.tx_start, exon.tx_end]
        if coding:
            boundaries.extend(
                value for value in (cds_start, cds_end) if exon.tx_start < value < exon.tx_end
            )
        boundaries = sorted(set(boundaries))
        for start, end in zip(boundaries, boundaries[1:]):
            if not coding:
                label = "noncoding_exon"
            elif end <= cds_start:
                label = "5putr"
            elif start >= cds_end:
                label = "3putr"
            else:
                label = "cds"
            labeled_exons.append(Region(start, end, label))

    result: list[Region] = []
    cursor = 0
    for region in labeled_exons:
        if cursor < region.start:
            result.append(Region(cursor, region.start, "intron"))
        result.append(region)
        cursor = region.end
    if cursor < tx.length:
        result.append(Region(cursor, tx.length, "intron"))
    return result


def annotate_regions(tx: Transcript) -> list[Region]:
    """Partition the selected coordinate space into exhaustive region labels."""

    cds_genomic = tx.feature_intervals.get("CDS", []) + tx.feature_intervals.get("stop_codon", [])
    cds = merge_intervals(
        piece
        for interval in cds_genomic
        for piece in tx.genomic_interval_to_transcript(*interval)
    )
    if tx.coordinate_space == "gene":
        return _gene_regions(tx, cds)
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
