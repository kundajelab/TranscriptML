"""Strict parsing of one-transcript-per-gene GENCODE-style GTF files."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from pathlib import Path

from transcriptml.rbpnet._progress import track
from transcriptml.rbpnet.coordinates import COORDINATE_SPACES, Exon, Transcript, annotate_regions

_ATTR_RE = re.compile(r'([^\s;]+)\s+(?:"([^"]*)"|([^;\s]+))')


def parse_attributes(text: str) -> dict[str, str]:
    """Parse GTF attributes into a string dictionary."""

    return {key: quoted or bare for key, quoted, bare in _ATTR_RE.findall(text)}


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open(encoding="utf-8")


def parse_gtf(
    path: str | Path,
    *,
    coordinate_space: str = "mature_transcript",
    progress: bool = True,
) -> list[Transcript]:
    """Parse and validate a one-transcript-per-gene annotation."""

    path = Path(path)
    if coordinate_space not in COORDINATE_SPACES:
        raise ValueError(
            f"coordinate_space must be one of {', '.join(COORDINATE_SPACES)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"GTF not found: {path}")
    transcripts: dict[str, Transcript] = {}
    exon_rows: dict[str, list[Exon]] = {}
    feature_rows: dict[str, dict[str, list[tuple[int, int]]]] = {}
    with _open_text(path) as handle:
        lines = track(handle, "rbpnet preprocess: parse GTF", unit="lines", enabled=progress)
        for line_no, line in enumerate(lines, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"{path}:{line_no}: expected 9 tab-separated GTF fields")
            chrom, _, feature, start_s, end_s, _, strand, _, attrs_s = fields
            attrs = parse_attributes(attrs_s)
            tx_id = attrs.get("transcript_id")
            if not tx_id:
                continue
            try:
                start, end = int(start_s) - 1, int(end_s)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid coordinates") from exc
            if feature == "transcript":
                if tx_id in transcripts:
                    raise ValueError(f"duplicate transcript row for {tx_id}")
                transcripts[tx_id] = Transcript(
                    transcript_id=tx_id,
                    gene_id=attrs.get("gene_id", ""),
                    gene_name=attrs.get("gene_name", ""),
                    transcript_name=attrs.get("transcript_name", ""),
                    transcript_type=attrs.get("transcript_type", attrs.get("gene_type", "")),
                    chrom=chrom,
                    strand=strand,
                    coordinate_space=coordinate_space,
                    genomic_start=start,
                    genomic_end=end,
                )
            elif feature == "exon":
                exon_rows.setdefault(tx_id, []).append(
                    Exon(
                        chrom,
                        start,
                        end,
                        strand,
                        attrs.get("exon_number", ""),
                        attrs.get("exon_id", ""),
                    )
                )
            elif feature in {"CDS", "UTR", "start_codon", "stop_codon"}:
                feature_rows.setdefault(tx_id, {}).setdefault(feature, []).append((start, end))
    if not transcripts:
        raise ValueError(f"no transcript records found in {path}")
    orphan_exons = set(exon_rows) - set(transcripts)
    if orphan_exons:
        raise ValueError(f"exons found without transcript row (example: {sorted(orphan_exons)[0]})")
    gene_counts = Counter(tx.gene_id for tx in transcripts.values() if tx.gene_id)
    duplicates = {gene_id for gene_id, count in gene_counts.items() if count > 1}
    if duplicates:
        raise ValueError(
            "annotation is not one-transcript-per-gene; multiple transcript rows found for "
            + sorted(duplicates)[0]
        )
    result: list[Transcript] = []
    for tx in transcripts.values():
        tx.exons = exon_rows.get(tx.transcript_id, [])
        tx.feature_intervals = feature_rows.get(tx.transcript_id, {})
        tx.finalize()
        tx.regions = annotate_regions(tx)
        result.append(tx)
    result.sort(key=lambda tx: (tx.chrom, min(e.start for e in tx.exons), tx.transcript_id))
    offset = 0
    for tx in result:
        tx.offset = offset
        offset += tx.length
    return result
