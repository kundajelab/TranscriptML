"""Strand-aware BAM crosslink extraction into a compact HDF5 matrix."""

from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import pysam

from transcriptml.rbpnet._progress import ProgressReporter, track
from transcriptml.rbpnet.coordinates import Transcript


def read1_rna_strand(is_reverse: bool, orientation: str) -> str | None:
    """Infer RNA strand from the read1 alignment and library convention."""

    read_strand = "-" if is_reverse else "+"
    if orientation == "opposite":
        return "+" if read_strand == "-" else "-"
    if orientation == "same":
        return read_strand
    if orientation == "unstranded":
        return None
    raise ValueError(f"unknown read1/RNA orientation: {orientation}")


def five_prime_reference_position(read) -> int | None:
    """Return the 5-prime aligned reference base, excluding soft clips."""

    if read.reference_start is None or read.reference_end is None:
        return None
    return read.reference_end - 1 if read.is_reverse else read.reference_start


def _alignment_is_compatible(
    read,
    exons: tuple[tuple[int, int], ...],
    introns: frozenset[tuple[int, int]],
) -> bool:
    """Check that the complete CIGAR is compatible with one mature transcript."""

    if read.reference_start is None or not read.cigartuples:
        return False
    reference_pos = read.reference_start
    has_aligned_bases = False

    def within_one_exon(start: int, end: int) -> bool:
        return any(exon_start <= start and end <= exon_end for exon_start, exon_end in exons)

    for operation, length in read.cigartuples:
        if length <= 0:
            return False
        if operation in {pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF}:
            if not within_one_exon(reference_pos, reference_pos + length):
                return False
            reference_pos += length
            has_aligned_bases = True
        elif operation == pysam.CDEL:
            if not within_one_exon(reference_pos, reference_pos + length):
                return False
            reference_pos += length
        elif operation == pysam.CREF_SKIP:
            if (reference_pos, reference_pos + length) not in introns:
                return False
            reference_pos += length
        elif operation in {pysam.CINS, pysam.CSOFT_CLIP, pysam.CHARD_CLIP, pysam.CPAD}:
            continue
        else:
            return False
    return has_aligned_bases


def _alignment_is_gene_compatible(read, start: int, end: int) -> bool:
    """Check that every reference-consuming CIGAR segment stays in one gene span.

    Full-gene coordinates contain introns, so ``N`` operations need not match
    the selected mature-transcript junctions. They do, however, have to remain
    completely within the selected gene locus.
    """

    if read.reference_start is None or not read.cigartuples:
        return False
    reference_pos = read.reference_start
    has_aligned_bases = False
    for operation, length in read.cigartuples:
        if length <= 0:
            return False
        if operation in {
            pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF, pysam.CDEL, pysam.CREF_SKIP,
        }:
            next_pos = reference_pos + length
            if reference_pos < start or next_pos > end:
                return False
            if operation in {pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF}:
                has_aligned_bases = True
            reference_pos = next_pos
        elif operation in {pysam.CINS, pysam.CSOFT_CLIP, pysam.CHARD_CLIP, pysam.CPAD}:
            continue
        else:
            return False
    return has_aligned_bases


def alignment_is_transcript_compatible(read, transcript: Transcript) -> bool:
    """Return whether an alignment is compatible with the selected coordinates."""

    if transcript.coordinate_space == "gene":
        assert transcript.genomic_start is not None and transcript.genomic_end is not None
        return _alignment_is_gene_compatible(
            read, transcript.genomic_start, transcript.genomic_end
        )

    exons = tuple(sorted((exon.start, exon.end) for exon in transcript.exons))
    introns = frozenset((left[1], right[0]) for left, right in zip(exons, exons[1:]))
    return _alignment_is_compatible(read, exons, introns)


class ExonBinIndex:
    """Small-memory genomic point index for candidate coordinate-space loci."""

    def __init__(self, transcripts: list[Transcript], bin_size: int = 16_384):
        self.transcripts = transcripts
        self.bin_size = bin_size
        self._bins: dict[tuple[str, str, int], list[tuple[int, int, int, int]]] = {}
        self._exons: list[tuple[tuple[int, int], ...]] = []
        self._introns: list[frozenset[tuple[int, int]]] = []
        self._gene_spans: list[tuple[int, int] | None] = []
        for tx_index, tx in enumerate(transcripts):
            genomic_exons = tuple(sorted((exon.start, exon.end) for exon in tx.exons))
            self._exons.append(genomic_exons)
            self._introns.append(frozenset(
                (left[1], right[0]) for left, right in zip(genomic_exons, genomic_exons[1:])
            ))
            if tx.coordinate_space == "gene":
                assert tx.genomic_start is not None and tx.genomic_end is not None
                self._gene_spans.append((tx.genomic_start, tx.genomic_end))
                intervals = [(tx.genomic_start, tx.genomic_end, 0)]
            else:
                self._gene_spans.append(None)
                intervals = [(exon.start, exon.end, exon.tx_start) for exon in tx.exons]
            for start, end, tx_start in intervals:
                record = (start, end, tx_index, tx_start)
                for bin_id in range(start // bin_size, (end - 1) // bin_size + 1):
                    self._bins.setdefault((tx.chrom, tx.strand, bin_id), []).append(record)

    def query(self, chrom: str, pos: int, strand: str | None) -> list[tuple[int, int]]:
        """Return ``(transcript_index, coordinate_position)`` locus hits."""

        strands = ("+", "-") if strand is None else (strand,)
        matches: list[tuple[int, int]] = []
        for query_strand in strands:
            for start, end, tx_index, tx_start in self._bins.get(
                (chrom, query_strand, pos // self.bin_size), []
            ):
                if start <= pos < end:
                    delta = pos - start if query_strand == "+" else end - 1 - pos
                    matches.append((tx_index, tx_start + delta))
        return matches

    def alignment_is_compatible(self, read, transcript_index: int) -> bool:
        """Check a read against the selected mature-transcript or gene span."""

        gene_span = self._gene_spans[transcript_index]
        if gene_span is not None:
            return _alignment_is_gene_compatible(read, *gene_span)
        return _alignment_is_compatible(
            read, self._exons[transcript_index], self._introns[transcript_index]
        )


def ensure_bam_index(path: str | Path) -> None:
    """Validate coordinate sorting and create a missing BAM index."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BAM not found: {path}")
    with pysam.AlignmentFile(str(path), "rb") as bam:
        sort_order = bam.header.to_dict().get("HD", {}).get("SO")
        if sort_order != "coordinate":
            raise ValueError(f"BAM must be coordinate sorted (SO:coordinate): {path}")
        try:
            bam.check_index()
            return
        except ValueError:
            pass
    try:
        pysam.index(str(path))
    except Exception as exc:
        raise RuntimeError(f"could not create BAM index for {path}: {exc}") from exc


def validate_bam_contigs(path: str | Path, transcripts: list[Transcript]) -> None:
    """Require every retained annotation contig to exist in a BAM header."""

    with pysam.AlignmentFile(str(path), "rb") as bam:
        missing = sorted({tx.chrom for tx in transcripts} - set(bam.references))
        if missing:
            raise ValueError(
                f"retained GTF chromosome(s) absent from BAM {path}: {', '.join(missing[:5])}; "
                f"BAM examples: {', '.join(bam.references[:5])}"
            )


def create_signal_store(
    path: str | Path,
    transcripts: list[Transcript],
    sample_names: list[str],
    sample_roles: list[str],
    *,
    compression: str | None = "gzip",
    compression_level: int | None = 1,
    chunk_length: int = 1_048_576,
) -> h5py.File:
    """Create the canonical concatenated locus-coordinate HDF5 store."""

    total_length = sum(tx.length for tx in transcripts)
    if total_length == 0:
        raise ValueError("annotation has zero coordinate-space bases")
    coordinate_spaces = {tx.coordinate_space for tx in transcripts}
    if len(coordinate_spaces) != 1:
        raise ValueError("all loci in one signal store must use the same coordinate space")
    coordinate_space = coordinate_spaces.pop()
    store = h5py.File(path, "w")
    store.attrs["format"] = "transcriptml-rbpnet-signals"
    store.attrs["format_version"] = "1"
    store.attrs["coordinate_space"] = coordinate_space
    store.attrs["coordinate_system"] = (
        "0-based half-open coordinates in annotated 5-prime to 3-prime orientation"
    )
    strings = h5py.string_dtype("utf-8")
    store.create_dataset(
        "transcript_ids",
        data=np.asarray([tx.transcript_id for tx in transcripts], dtype=object),
        dtype=strings,
    )
    store.create_dataset("transcript_offsets", data=np.asarray([tx.offset for tx in transcripts], dtype=np.int64))
    store.create_dataset("transcript_lengths", data=np.asarray([tx.length for tx in transcripts], dtype=np.int64))
    store.create_dataset("sample_names", data=np.asarray(sample_names, dtype=object), dtype=strings)
    store.create_dataset("sample_roles", data=np.asarray(sample_roles, dtype=object), dtype=strings)
    compression, compression_opts = _normalize_compression(
        compression, compression_level
    )
    chunk = min(total_length, int(chunk_length))
    if chunk <= 0:
        raise ValueError("chunk_length must be positive")
    store.attrs["signal_compression"] = "none" if compression is None else compression
    store.attrs["signal_compression_level"] = (
        -1 if compression_opts is None else int(compression_opts)
    )
    store.attrs["signal_chunk_length"] = chunk
    store.create_dataset(
        "counts",
        shape=(len(sample_names), total_length),
        dtype=np.uint32,
        chunks=(1, chunk),
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
        fillvalue=0,
    )
    return store


def _normalize_compression(
    compression: str | None,
    compression_level: int | None,
) -> tuple[str | None, int | None]:
    """Validate signal compression and return h5py keyword values."""

    if compression is None or str(compression).strip().lower() in {
        "none", "off", "uncompressed",
    }:
        if compression_level not in {None, 0}:
            raise ValueError("compression_level is only valid with gzip")
        return None, None
    name = str(compression).strip().lower()
    if name == "gzip":
        level = 1 if compression_level is None else int(compression_level)
        if level < 0 or level > 9:
            raise ValueError("gzip compression_level must be between 0 and 9")
        return name, level
    if name == "lzf":
        if compression_level not in {None, 0}:
            raise ValueError("lzf does not accept a compression level")
        return name, None
    raise ValueError("signal compression must be one of: gzip, lzf, none")


def _sqlite_sparse_batches(
    cursor: sqlite3.Cursor,
    *,
    batch_size: int = 100_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield sorted sparse position/count arrays from the aggregation table."""

    while True:
        rows = cursor.fetchmany(int(batch_size))
        if not rows:
            return
        positions = np.fromiter(
            (item[0] for item in rows), dtype=np.int64, count=len(rows)
        )
        values = np.fromiter(
            (item[1] for item in rows), dtype=np.uint64, count=len(rows)
        )
        yield positions, values


def write_sparse_signal_chunkwise(
    dataset: h5py.Dataset,
    row: int,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    reporter: ProgressReporter | None = None,
) -> int:
    """Write a sorted sparse signal row using one dense write per HDF5 chunk.

    The input positions must be unique and strictly increasing across batches.
    Only chunks containing at least one nonzero count are materialized. A
    dense ``uint32`` buffer is retained when a sparse input batch ends partway
    through a chunk, ensuring that even such chunks are written exactly once.
    """

    if dataset.ndim != 2 or dataset.chunks is None:
        raise ValueError("signal dataset must be a two-dimensional chunked dataset")
    if row < 0 or row >= dataset.shape[0]:
        raise IndexError("signal dataset row is outside bounds")
    row_chunk, chunk_length = (int(value) for value in dataset.chunks)
    if row_chunk != 1:
        raise ValueError("signal dataset must use one sample row per HDF5 chunk")
    total_length = int(dataset.shape[1])
    uint32_max = np.iinfo(np.uint32).max
    active_chunk = -1
    active_start = 0
    active_buffer: np.ndarray | None = None
    previous_position = -1
    written_positions = 0

    def flush() -> None:
        nonlocal active_buffer
        if active_buffer is None:
            return
        dataset[row, active_start : active_start + active_buffer.size] = active_buffer
        active_buffer = None

    for raw_positions, raw_values in batches:
        positions = np.asarray(raw_positions, dtype=np.int64)
        values = np.asarray(raw_values, dtype=np.uint64)
        if positions.ndim != 1 or values.ndim != 1 or positions.shape != values.shape:
            raise ValueError("sparse signal positions and values must be aligned vectors")
        if positions.size == 0:
            continue
        if positions[0] <= previous_position or np.any(np.diff(positions) <= 0):
            raise ValueError("sparse signal positions must be unique and strictly increasing")
        if positions[0] < 0 or positions[-1] >= total_length:
            raise IndexError("sparse signal position is outside dataset bounds")
        if np.any(values == 0):
            raise ValueError("sparse signal values must be positive")
        if values.max(initial=0) > uint32_max:
            raise OverflowError("a crosslink-position count exceeds uint32")

        offset = 0
        while offset < positions.size:
            chunk_index = int(positions[offset] // chunk_length)
            chunk_end_position = min((chunk_index + 1) * chunk_length, total_length)
            end = int(np.searchsorted(positions, chunk_end_position, side="left"))
            if chunk_index != active_chunk:
                flush()
                active_chunk = chunk_index
                active_start = chunk_index * chunk_length
                active_buffer = np.zeros(
                    chunk_end_position - active_start, dtype=np.uint32
                )
            assert active_buffer is not None
            local_positions = positions[offset:end] - active_start
            active_buffer[local_positions] = values[offset:end].astype(
                np.uint32, copy=False
            )
            written = end - offset
            written_positions += written
            if reporter is not None:
                reporter.update(written)
            offset = end
        previous_position = int(positions[-1])
    flush()
    return written_positions


def _flush_counts(connection: sqlite3.Connection, counts: Counter[int]) -> None:
    if not counts:
        return
    connection.executemany(
        "INSERT INTO counts(position, count) VALUES (?, ?) "
        "ON CONFLICT(position) DO UPDATE SET count=count+excluded.count",
        counts.items(),
    )
    connection.commit()
    counts.clear()


def extract_bam_to_store(
    bam_path: str | Path,
    row: int,
    dataset: h5py.Dataset,
    transcripts: list[Transcript],
    exon_index: ExonBinIndex,
    orientation: str,
    min_mapq: int,
    exclude_duplicates: bool,
    temp_dir: str | Path | None = None,
    *,
    progress: bool = True,
) -> tuple[dict, np.ndarray]:
    """Stream one BAM, disk-aggregate sparse events, and fill one HDF5 row."""

    ensure_bam_index(bam_path)
    validate_bam_contigs(bam_path, transcripts)
    qc: Counter[str] = Counter()
    transcript_counts = np.zeros(len(transcripts), dtype=np.int64)
    transcript_hit = np.zeros(len(transcripts), dtype=bool)
    with tempfile.TemporaryDirectory(prefix="transcriptml_rbpnet_", dir=temp_dir) as work:
        connection = sqlite3.connect(str(Path(work) / "counts.sqlite"))
        connection.execute("CREATE TABLE counts(position INTEGER PRIMARY KEY, count INTEGER NOT NULL)")
        batch: Counter[int] = Counter()
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            try:
                total_records = bam.mapped + bam.unmapped
            except (AttributeError, ValueError):
                total_records = None
            reads = track(
                bam.fetch(until_eof=True),
                f"rbpnet preprocess: read {Path(bam_path).name}",
                total=total_records,
                unit="records",
                enabled=progress,
            )
            for read in reads:
                qc["records_seen"] += 1
                if not read.is_read1:
                    qc["not_read1"] += 1
                    continue
                qc["read1_seen"] += 1
                if read.is_unmapped:
                    qc["unmapped"] += 1
                    continue
                if read.is_secondary:
                    qc["secondary"] += 1
                    continue
                if read.is_supplementary:
                    qc["supplementary"] += 1
                    continue
                if read.is_qcfail:
                    qc["qc_fail"] += 1
                    continue
                if exclude_duplicates and read.is_duplicate:
                    qc["duplicate"] += 1
                    continue
                if read.mapping_quality < min_mapq:
                    qc["low_mapq"] += 1
                    continue
                pos = five_prime_reference_position(read)
                if pos is None:
                    qc["invalid_crosslink_position"] += 1
                    continue
                qc["passing_filters"] += 1
                strand = read1_rna_strand(read.is_reverse, orientation)
                matches = exon_index.query(read.reference_name, pos, strand)
                if not matches:
                    qc["no_compatible_transcript"] += 1
                    continue
                compatible = [m for m in matches if exon_index.alignment_is_compatible(read, m[0])]
                if not compatible:
                    qc["transcript_incompatible"] += 1
                    continue
                if len(compatible) != 1 or len({tx_index for tx_index, _ in compatible}) != 1:
                    qc["ambiguous_transcript"] += 1
                    continue
                tx_index, tx_pos = compatible[0]
                flat_position = transcripts[tx_index].offset + tx_pos
                batch[flat_position] += 1
                transcript_counts[tx_index] += 1
                transcript_hit[tx_index] = True
                qc["retained"] += 1
                if len(batch) >= 100_000:
                    _flush_counts(connection, batch)
        _flush_counts(connection, batch)
        unique_total = int(connection.execute("SELECT COUNT(*) FROM counts").fetchone()[0])
        cursor = connection.execute("SELECT position, count FROM counts ORDER BY position")
        unique_positions = 0
        reporter = ProgressReporter(
            f"rbpnet preprocess: write {Path(bam_path).name} signal",
            total=unique_total,
            unit="positions",
            enabled=progress,
        )
        try:
            unique_positions = write_sparse_signal_chunkwise(
                dataset,
                row,
                _sqlite_sparse_batches(cursor),
                reporter=reporter,
            )
        except OverflowError as exc:
            raise OverflowError(
                f"a crosslink-position count in {bam_path} exceeds uint32"
            ) from exc
        reporter.close()
        connection.close()
    qc["unique_crosslink_positions"] = unique_positions
    qc["transcripts_with_signal"] = int(transcript_hit.sum())
    qc["transcripts_total"] = len(transcripts)
    fields = (
        "records_seen", "read1_seen", "not_read1", "unmapped", "secondary", "supplementary",
        "qc_fail", "duplicate", "low_mapq", "invalid_crosslink_position", "passing_filters",
        "no_compatible_transcript", "transcript_incompatible", "ambiguous_transcript", "retained",
        "unique_crosslink_positions", "transcripts_with_signal", "transcripts_total",
    )
    return {field: int(qc[field]) for field in fields}, transcript_counts


def write_ip_pooled(store: h5py.File, ip_rows: list[int], *, progress: bool = True) -> None:
    """Sum IP rows chunkwise into the derived pooled-IP HDF5 track."""

    counts = store["counts"]
    total = counts.shape[1]
    chunk = counts.chunks[1]
    compression = counts.compression
    compression_opts = counts.compression_opts if compression == "gzip" else None
    pooled = store.create_dataset(
        "ip_pooled",
        shape=(total,),
        dtype=np.uint32,
        chunks=(chunk,),
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
        fillvalue=0,
    )
    starts = range(0, total, chunk)
    for start in track(
        starts,
        "rbpnet preprocess: pool IP tracks",
        total=len(starts),
        unit="chunks",
        enabled=progress,
    ):
        end = min(start + chunk, total)
        values = counts[ip_rows, start:end].astype(np.uint64).sum(axis=0)
        if values.max(initial=0) > np.iinfo(np.uint32).max:
            raise OverflowError("pooled IP count exceeds uint32")
        if np.any(values):
            pooled[start:end] = values.astype(np.uint32)
    pooled.attrs["source_rows"] = np.asarray(ip_rows, dtype=np.int64)
