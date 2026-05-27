from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from transcriptml.data.encoding import DEFAULT_SALUKI_LENGTH
from transcriptml.progress import ProgressReporter, log_progress


_GTF_ATTR_RE = re.compile(r'\s*([^\s=;]+)\s+(?:"([^"]*)"|([^;]*))\s*;?')
_DNA_COMPLEMENT = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn", "TGCAYRKMVHDBNtgcayrkmvhdbn")


@dataclass(frozen=True)
class GTFRecord:
    """A single GTF/GFF-like feature row with 0-based half-open coordinates."""

    chrom: str
    source: str
    feature: str
    start: int
    end: int
    score: str
    strand: str
    frame: str
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class TranscriptFeature:
    """Genomic annotation needed to build one transcript."""

    transcript_id: str
    chrom: str
    strand: str
    exons: tuple[GTFRecord, ...]
    cds: tuple[GTFRecord, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    @property
    def exon_count(self) -> int:
        """Return the number of exon features in the transcript."""

        return len(self.exons)

    @property
    def transcript_length(self) -> int:
        """Return total spliced transcript length in nucleotides."""

        return int(sum(r.end - r.start for r in self.exons))


@dataclass(frozen=True)
class TranscriptRecord:
    """A transcript sequence plus transcript-coordinate annotation channels."""

    transcript_id: str
    sequence: str
    cds_positions: tuple[int, ...]
    splice_positions: tuple[int, ...]
    metadata: Mapping[str, object]


@dataclass
class _TranscriptMutable:
    transcript_id: str
    chrom: str
    strand: str
    attrs: dict[str, str]
    exons: list[GTFRecord] = field(default_factory=list)
    cds: list[GTFRecord] = field(default_factory=list)


@dataclass(frozen=True)
class _ExonMap:
    record: GTFRecord
    tx_start: int
    tx_end: int


class _FastaAccessor:
    def __init__(self, path: str | Path):
        """Open a FASTA through pyfaidx or a small in-memory fallback."""

        self.path = Path(path)
        self._fa = None
        self._seqs: dict[str, str] | None = None
        try:
            from pyfaidx import Fasta
        except ImportError:
            self._seqs = _read_fasta_into_memory(self.path)
            self._keys = set(self._seqs)
        else:
            self._fa = Fasta(str(self.path), as_raw=True, sequence_always_upper=True)
            self._keys = {str(k) for k in self._fa.keys()}

    @property
    def keys(self) -> set[str]:
        """Return available FASTA sequence names."""

        return self._keys

    def has_chrom(self, chrom: str) -> bool:
        """Return whether a chromosome can be resolved in the FASTA."""

        return _resolve_chrom_name(chrom, self.keys) is not None

    def close(self) -> None:
        """Close the underlying FASTA handle when one exists."""

        if self._fa is not None and hasattr(self._fa, "close"):
            self._fa.close()

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch an uppercase genomic interval using 0-based half-open coordinates."""

        key = _resolve_chrom_name(chrom, self.keys)
        if key is None:
            raise KeyError(f"Chromosome '{chrom}' was not found in FASTA {self.path}")
        if self._fa is not None:
            seq = self._fa[key][int(start) : int(end)]
            return str(seq if isinstance(seq, str) else seq.seq).upper()
        assert self._seqs is not None
        return self._seqs[key][int(start) : int(end)].upper()


def _open_text(path: str | Path):
    """Open plain or gzip-compressed text for reading."""

    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open("r", encoding="utf-8")


def _read_fasta_into_memory(path: Path) -> dict[str, str]:
    """Read a small FASTA file into an uppercase sequence dictionary."""

    seqs: dict[str, list[str]] = {}
    current: str | None = None
    with _open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs.setdefault(current, [])
            elif current is None:
                raise ValueError(f"FASTA sequence encountered before header in {path}")
            else:
                seqs[current].append(line)
    return {name: "".join(parts).upper() for name, parts in seqs.items()}


def _resolve_chrom_name(chrom: str, keys: set[str]) -> str | None:
    """Resolve exact and common ``chr``/no-``chr`` chromosome aliases."""

    if chrom in keys:
        return chrom
    candidates = []
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")
    if chrom in {"M", "MT"}:
        candidates.extend(["chrM", "chrMT"])
    if chrom == "chrM":
        candidates.extend(["M", "MT"])
    for cand in candidates:
        if cand in keys:
            return cand
    return None


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    return str(seq).translate(_DNA_COMPLEMENT)[::-1]


def parse_gtf_attributes(text: str) -> dict[str, str]:
    """Parse GTF attributes without depending on pyranges/rtracklayer.

    The parser accepts canonical GTF attributes such as
    ``gene_id "G"; transcript_id "T";`` and common GFF3-style ``key=value``
    attributes found in converted files.
    """

    attrs: dict[str, str] = {}
    text = str(text).strip()
    if not text:
        return attrs
    if "=" in text and '"' not in text:
        for part in text.rstrip(";").split(";"):
            if not part.strip() or "=" not in part:
                continue
            key, value = part.split("=", 1)
            attrs[key.strip()] = value.strip()
        return attrs
    for match in _GTF_ATTR_RE.finditer(text):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        attrs[key] = str(value).strip()
    if attrs:
        return attrs
    for part in text.rstrip(";").split(";"):
        tokens = part.strip().split(None, 1)
        if len(tokens) == 2:
            attrs[tokens[0]] = tokens[1].strip().strip('"')
    return attrs


def iter_gtf_records(
    path: str | Path,
    *,
    features: Sequence[str] | None = None,
) -> Iterator[GTFRecord]:
    """Yield GTF feature records using 0-based half-open coordinates."""

    keep = {f.lower() for f in features} if features is not None else None
    with _open_text(path) as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                raise ValueError(f"Expected 9 GTF columns at {path}:{line_no}, found {len(fields)}")
            feature = fields[2]
            if keep is not None and feature.lower() not in keep:
                continue
            try:
                start = int(fields[3]) - 1
                end = int(fields[4])
            except ValueError as exc:
                raise ValueError(f"Invalid GTF coordinates at {path}:{line_no}") from exc
            if start < 0 or end < start:
                raise ValueError(f"Invalid GTF interval at {path}:{line_no}: {fields[3]}-{fields[4]}")
            yield GTFRecord(
                chrom=fields[0],
                source=fields[1],
                feature=feature,
                start=start,
                end=end,
                score=fields[5],
                strand=fields[6],
                frame=fields[7],
                attributes=parse_gtf_attributes(fields[8]),
            )


def load_transcript_features(
    gtf_path: str | Path,
    *,
    transcript_ids: set[str] | Sequence[str] | None = None,
    progress: bool = True,
) -> dict[str, TranscriptFeature]:
    """Load exon/CDS structures from a GTF without using pyranges."""

    wanted = {str(x) for x in transcript_ids} if transcript_ids is not None else None
    grouped: dict[str, _TranscriptMutable] = {}
    reporter = ProgressReporter("parse GTF exon/CDS records", unit="records", enabled=progress)
    for rec in iter_gtf_records(gtf_path, features=("exon", "CDS")):
        reporter.update()
        tid = rec.attributes.get("transcript_id") or rec.attributes.get("transcript")
        if not tid or (wanted is not None and tid not in wanted):
            continue
        if rec.strand not in {"+", "-"}:
            raise ValueError(f"Transcript {tid} has unsupported strand '{rec.strand}'")
        item = grouped.get(tid)
        if item is None:
            item = _TranscriptMutable(
                transcript_id=tid,
                chrom=rec.chrom,
                strand=rec.strand,
                attrs=dict(rec.attributes),
            )
            grouped[tid] = item
        elif item.chrom != rec.chrom or item.strand != rec.strand:
            raise ValueError(f"Transcript {tid} has features on multiple chromosomes or strands")
        item.attrs.update({k: v for k, v in rec.attributes.items() if k not in item.attrs})
        if rec.feature.lower() == "exon":
            item.exons.append(rec)
        elif rec.feature.lower() == "cds":
            item.cds.append(rec)

    out: dict[str, TranscriptFeature] = {}
    for tid, item in grouped.items():
        if not item.exons:
            continue
        out[tid] = TranscriptFeature(
            transcript_id=tid,
            chrom=item.chrom,
            strand=item.strand,
            exons=tuple(item.exons),
            cds=tuple(item.cds),
            attributes=dict(item.attrs),
        )
    reporter.close(extra=f"{len(out)} transcripts with exons")
    return out


def _transcript_ordered_exons(feature: TranscriptFeature) -> list[GTFRecord]:
    """Return exons ordered in transcript orientation."""

    if feature.strand == "+":
        return sorted(feature.exons, key=lambda r: (r.start, r.end))
    if feature.strand == "-":
        return sorted(feature.exons, key=lambda r: (r.end, r.start), reverse=True)
    raise ValueError(f"Transcript {feature.transcript_id} has unsupported strand '{feature.strand}'")


def _build_exon_map(feature: TranscriptFeature) -> list[_ExonMap]:
    """Build transcript-coordinate spans for each ordered exon."""

    offset = 0
    mapped: list[_ExonMap] = []
    for exon in _transcript_ordered_exons(feature):
        length = int(exon.end - exon.start)
        mapped.append(_ExonMap(record=exon, tx_start=offset, tx_end=offset + length))
        offset += length
    return mapped


def _sequence_for_feature(feature: TranscriptFeature, fasta: _FastaAccessor) -> str:
    """Assemble a spliced transcript sequence from genomic exon intervals."""

    parts: list[str] = []
    for exon in _transcript_ordered_exons(feature):
        seq = fasta.fetch(exon.chrom, exon.start, exon.end)
        parts.append(reverse_complement(seq) if feature.strand == "-" else seq)
    return "".join(parts).upper()


def _map_interval_to_transcript(
    interval: GTFRecord,
    exon_map: Sequence[_ExonMap],
    *,
    strand: str,
) -> list[tuple[int, int]]:
    """Map a genomic interval onto transcript-coordinate intervals."""

    mapped: list[tuple[int, int]] = []
    for exon in exon_map:
        ex = exon.record
        ov_start = max(int(interval.start), int(ex.start))
        ov_end = min(int(interval.end), int(ex.end))
        if ov_start >= ov_end:
            continue
        if strand == "+":
            tx_start = exon.tx_start + (ov_start - ex.start)
            tx_end = exon.tx_start + (ov_end - ex.start)
        else:
            tx_start = exon.tx_start + (ex.end - ov_end)
            tx_end = exon.tx_start + (ex.end - ov_start)
        mapped.append((int(tx_start), int(tx_end)))
    return mapped


def _cds_codon_positions(feature: TranscriptFeature, exon_map: Sequence[_ExonMap]) -> tuple[int, ...]:
    """Return transcript-coordinate codon-start positions for CDS features."""

    ranges: list[tuple[int, int]] = []
    for cds in feature.cds:
        ranges.extend(_map_interval_to_transcript(cds, exon_map, strand=feature.strand))
    if not ranges:
        return ()
    cds_start = min(start for start, _ in ranges)
    cds_end = max(end for _, end in ranges)
    return tuple(int(x) for x in range(int(cds_start), int(cds_end), 3))


def transcript_record_from_feature(feature: TranscriptFeature, fasta: _FastaAccessor) -> TranscriptRecord:
    """Create a transcript sequence record from one annotated transcript feature."""

    exon_map = _build_exon_map(feature)
    sequence = _sequence_for_feature(feature, fasta)
    splice_positions = tuple(int(exon.tx_start - 1) for exon in exon_map[1:])
    cds_positions = _cds_codon_positions(feature, exon_map)
    attrs = dict(feature.attributes)
    metadata: dict[str, object] = {
        "chrom": feature.chrom,
        "strand": feature.strand,
        "exon_count": feature.exon_count,
        "transcript_length": len(sequence),
        "cds_length": len(cds_positions) * 3 if cds_positions else 0,
    }
    for key in ("gene_id", "gene_name", "transcript_name", "gene_type", "transcript_type"):
        if key in attrs:
            metadata[key] = attrs[key]
    return TranscriptRecord(
        transcript_id=feature.transcript_id,
        sequence=sequence,
        cds_positions=cds_positions,
        splice_positions=splice_positions,
        metadata=metadata,
    )


def extract_transcript_records(
    gtf_path: str | Path,
    fasta_path: str | Path,
    *,
    transcript_ids: set[str] | Sequence[str] | None = None,
    progress: bool = True,
) -> list[TranscriptRecord]:
    """Extract transcript sequences and Saluki annotation positions from GTF/FASTA."""

    features = load_transcript_features(gtf_path, transcript_ids=transcript_ids, progress=progress)
    log_progress(f"opening FASTA {fasta_path}", enabled=progress)
    fasta = _FastaAccessor(fasta_path)
    try:
        reporter = ProgressReporter(
            "extract transcript records",
            total=len(features),
            unit="transcripts",
            enabled=progress,
        )
        records = []
        for feature in features.values():
            records.append(transcript_record_from_feature(feature, fasta))
            reporter.update()
        reporter.close()
        return records
    finally:
        fasta.close()


def write_saluki_memmap(
    path: str | Path,
    records: Sequence[TranscriptRecord],
    *,
    length: int = DEFAULT_SALUKI_LENGTH,
    dtype: np.dtype | type = np.uint8,
    progress: bool = True,
) -> np.memmap:
    """Encode transcript records to a Saluki ``X.npy`` file without a RAM-sized copy."""

    from transcriptml.data.encoding import encode_saluki_transcript

    X = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(records), 6, int(length)))
    reporter = ProgressReporter("encode Saluki transcripts", total=len(records), unit="transcripts", enabled=progress)
    for i, record in enumerate(records):
        X[i] = encode_saluki_transcript(
            record.sequence,
            length=int(length),
            cds_positions=record.cds_positions,
            splice_positions=record.splice_positions,
            dtype=dtype,
        )
        reporter.update()
    X.flush()
    reporter.close()
    return X
