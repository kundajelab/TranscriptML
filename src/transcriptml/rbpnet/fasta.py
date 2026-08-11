"""FASTA validation and transcript-oriented locus sequence extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pysam

from transcriptml.rbpnet._progress import track
from transcriptml.rbpnet.coordinates import Transcript

_COMPLEMENT = str.maketrans("ACGTRYMKBDHVacgtrymkbdhv", "TGCAYRKMVHDBtgcayrkmvhdb")


def ensure_fasta_index(path: str | Path) -> Path:
    """Return a FASTA index path, creating the index when reasonable."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA not found: {path}")
    index = Path(str(path) + ".fai")
    if not index.exists():
        try:
            pysam.faidx(str(path))
        except Exception as exc:
            raise RuntimeError(f"could not create FASTA index {index}: {exc}") from exc
    return index


def reverse_complement(sequence: str) -> str:
    """Return the IUPAC-aware DNA reverse complement."""

    return sequence.translate(_COMPLEMENT)[::-1]


def retain_fasta_transcripts(
    genome_fasta: str | Path, transcripts: list[Transcript]
) -> tuple[list[Transcript], dict]:
    """Skip GTF transcripts on contigs intentionally absent from the FASTA."""

    ensure_fasta_index(genome_fasta)
    with pysam.FastaFile(str(genome_fasta)) as fasta:
        fasta_contigs = set(fasta.references)
    retained = [tx for tx in transcripts if tx.chrom in fasta_contigs]
    missing_counts = Counter(tx.chrom for tx in transcripts if tx.chrom not in fasta_contigs)
    if not retained:
        missing = ", ".join(sorted(missing_counts)[:5]) or "none"
        raise ValueError(
            "no annotated transcripts remain after intersecting GTF and FASTA contigs; "
            f"GTF-only contig examples: {missing}"
        )
    offset = 0
    for tx in retained:
        tx.offset = offset
        offset += tx.length
    return retained, {
        "gtf_transcripts_total": len(transcripts),
        "transcripts_retained": len(retained),
        "transcripts_skipped_missing_fasta_contig": len(transcripts) - len(retained),
        "gtf_contigs_missing_from_fasta": dict(sorted(missing_counts.items())),
    }


def transcript_sequence(fasta: pysam.FastaFile, tx: Transcript) -> str:
    """Extract one selected locus in annotated RNA 5-prime to 3-prime order."""

    if tx.coordinate_space == "gene":
        assert tx.genomic_start is not None and tx.genomic_end is not None
        sequence = fasta.fetch(tx.chrom, tx.genomic_start, tx.genomic_end)
        if tx.strand == "-":
            sequence = reverse_complement(sequence)
    else:
        chunks = []
        for exon in tx.exons:
            chunk = fasta.fetch(exon.chrom, exon.start, exon.end)
            chunks.append(chunk if tx.strand == "+" else reverse_complement(chunk))
        sequence = "".join(chunks)
    sequence = sequence.upper()
    if len(sequence) != tx.length:
        raise RuntimeError(f"sequence length mismatch for {tx.transcript_id}")
    return sequence


def write_transcript_fasta(
    path: str | Path,
    genome_fasta: str | Path,
    transcripts: list[Transcript],
    *,
    progress: bool = True,
) -> dict:
    """Write indexed transcript-oriented locus FASTA and return sequence QC."""

    path = Path(path)
    ensure_fasta_index(genome_fasta)
    stats = {"transcripts": len(transcripts), "bases": 0, "non_acgtn_bases": 0}
    with pysam.FastaFile(str(genome_fasta)) as fasta, path.open("w", encoding="utf-8") as out:
        missing = sorted({tx.chrom for tx in transcripts} - set(fasta.references))
        if missing:
            raise ValueError(
                f"GTF chromosome(s) absent from FASTA: {', '.join(missing[:5])}; "
                f"FASTA examples: {', '.join(fasta.references[:5])}"
            )
        for tx in track(
            transcripts,
            "rbpnet preprocess: extract transcript sequences",
            total=len(transcripts),
            unit="transcripts",
            enabled=progress,
        ):
            sequence = transcript_sequence(fasta, tx)
            stats["bases"] += len(sequence)
            stats["non_acgtn_bases"] += sum(base not in "ACGTN" for base in sequence)
            out.write(f">{tx.transcript_id}\n")
            for start in range(0, len(sequence), 80):
                out.write(sequence[start : start + 80] + "\n")
    pysam.faidx(str(path))
    return stats
