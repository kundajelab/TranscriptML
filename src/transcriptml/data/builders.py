from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from transcriptml.data.bundle import DatasetBundle, save_bundle, save_bundle_metadata
from transcriptml.data.encoding import DEFAULT_SALUKI_LENGTH, encode_rna_sequence, encode_saluki_transcript
from transcriptml.data.genomics import _FastaAccessor, load_transcript_features, transcript_record_from_feature
from transcriptml.data.schemas import RNA4, SALUKI6
from transcriptml.progress import ProgressReporter, log_progress


def _infer_delimiter(path: str | Path, delimiter: str | None) -> str:
    """Return the explicit delimiter or infer CSV/TSV from the filename.

    Args:
        path: Source table path whose suffix is used for delimiter inference.
        delimiter: Explicit delimiter to use, or ``None`` to infer from
            ``path``.
    """

    if delimiter is not None:
        return delimiter
    return "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","


def _read_rows(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, str]]:
    """Read a delimited table into dictionaries keyed by header names.

    Args:
        path: CSV/TSV-like table with a header row.
        delimiter: Optional delimiter override. When ``None``, the delimiter is
            inferred from ``path``.
    """

    delim = _infer_delimiter(path, delimiter)
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return [dict(row) for row in reader]


def _metadata_for_row(row: Mapping[str, str], exclude: set[str], metadata_cols: Sequence[str] | None) -> dict[str, Any]:
    """Select metadata fields from a source row.

    Args:
        row: Source table row keyed by column name.
        exclude: Column names that should not be copied into metadata when
            ``metadata_cols`` is not provided.
        metadata_cols: Optional explicit column names to keep as metadata.
    """

    cols = metadata_cols if metadata_cols is not None else [c for c in row if c not in exclude]
    return {c: row.get(c) for c in cols if c in row}


def _parse_positions(value: str | None) -> list[int]:
    """Parse transcript-coordinate positions from JSON or delimiter-separated text.

    Args:
        value: Optional string containing positions as a JSON list or comma,
            semicolon, or pipe-separated integers.
    """

    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        loaded = json.loads(text)
        return [int(x) for x in loaded]
    for sep in (";", "|"):
        text = text.replace(sep, ",")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _dedupe_target_rows(rows: Sequence[Mapping[str, str]], id_col: str) -> dict[str, Mapping[str, str]]:
    """Index target rows by transcript id and reject duplicate ids.

    Args:
        rows: Target table rows keyed by column name.
        id_col: Column containing the transcript identifier for each row.
    """

    out: dict[str, Mapping[str, str]] = {}
    for row in rows:
        tid = str(row[id_col])
        if tid in out:
            raise ValueError(f"Duplicate target row for transcript id '{tid}'")
        out[tid] = row
    return out


def _splits_from_rows(rows: Sequence[Mapping[str, str]], split_col: str | None) -> dict[str, list[int]] | None:
    """Build train/validation/test index lists from a table split column.

    Args:
        rows: Source rows whose order defines dataset indices.
        split_col: Optional column containing split labels such as ``train``,
            ``val``, ``valid``, ``validation``, or ``test``.
    """

    if split_col is None:
        return None
    splits = {"train": [], "val": [], "test": []}
    val_aliases = {"val", "valid", "validation"}
    for i, row in enumerate(rows):
        value = str(row.get(split_col, "")).strip().lower()
        if value == "train":
            splits["train"].append(i)
        elif value in val_aliases:
            splits["val"].append(i)
        elif value == "test":
            splits["test"].append(i)
    return splits


def build_mpra_dataset(
    table_path: str | Path,
    out_dir: str | Path,
    *,
    sequence_col: str,
    target_col: str | None = None,
    id_col: str | None = None,
    length: int | None = None,
    metadata_cols: Sequence[str] | None = None,
    split_col: str | None = None,
    delimiter: str | None = None,
    progress: bool = True,
) -> DatasetBundle:
    """Build an RNA4 MPRA dataset from a delimited table.

    Args:
        table_path: Input CSV/TSV-like table containing sequences and optional
            targets.
        out_dir: Directory where the processed dataset bundle is written.
        sequence_col: Column containing RNA or DNA sequence strings.
        target_col: Optional column containing scalar regression targets.
        id_col: Optional column containing stable example identifiers. Row
            indices are used when omitted.
        length: Fixed encoded sequence length. When ``None``, the longest input
            sequence length is used.
        metadata_cols: Optional columns to copy into bundle metadata. When
            omitted, non-sequence, non-target, and non-id columns are kept.
        split_col: Optional column with train/validation/test split labels.
        delimiter: Optional table delimiter override.
        progress: Whether to emit progress messages while building the bundle.
    """

    log_progress(f"build-mpra: reading {table_path}", enabled=progress)
    rows = _read_rows(table_path, delimiter=delimiter)
    seqs = [row[sequence_col] for row in rows]
    if length is None:
        length = max((len(str(s)) for s in seqs), default=0)
    if length <= 0:
        raise ValueError("Cannot encode an empty collection without a positive length")
    X = np.zeros((len(seqs), 4, int(length)), dtype=np.uint8)
    reporter = ProgressReporter("build-mpra: encode sequences", total=len(seqs), unit="sequences", enabled=progress)
    for i, seq in enumerate(seqs):
        X[i] = encode_rna_sequence(seq, length=int(length))
        reporter.update()
    reporter.close()
    y = None
    if target_col is not None:
        log_progress("build-mpra: reading targets", enabled=progress)
        y = np.array([float(row[target_col]) for row in rows], dtype=np.float32)
    ids = [str(row[id_col]) if id_col else str(i) for i, row in enumerate(rows)]
    exclude = {sequence_col}
    if target_col:
        exclude.add(target_col)
    if id_col:
        exclude.add(id_col)
    metadata = [_metadata_for_row(row, exclude, metadata_cols) for row in rows]
    bundle = DatasetBundle(
        X=X,
        y=y,
        ids=ids,
        schema=RNA4,
        metadata=metadata,
        splits=_splits_from_rows(rows, split_col),
        config={
            "builder": "mpra",
            "source": str(table_path),
            "sequence_col": sequence_col,
            "target_col": target_col,
            "id_col": id_col,
            "split_col": split_col,
            "length": int(length),
        },
    )
    log_progress(f"build-mpra: saving bundle to {out_dir}", enabled=progress)
    save_bundle(bundle, out_dir)
    log_progress("build-mpra: done", enabled=progress)
    return bundle


def build_saluki_dataset(
    *,
    table_path: str | Path,
    out_dir: str | Path,
    sequence_col: str,
    id_col: str,
    target_col: str | None = None,
    cds_positions_col: str | None = None,
    splice_positions_col: str | None = None,
    length: int = DEFAULT_SALUKI_LENGTH,
    metadata_cols: Sequence[str] | None = None,
    split_col: str | None = None,
    delimiter: str | None = None,
    progress: bool = True,
) -> DatasetBundle:
    """Build a Saluki-style fixed-length ``(N, 6, L)`` transcript dataset.

    This table-based builder expects transcript sequences and optional
    transcript-coordinate annotation positions. Use
    :func:`build_saluki_dataset_from_gtf` when starting from a genome FASTA and
    transcript annotation GTF.

    Args:
        table_path: Input CSV/TSV-like table containing transcript sequences.
        out_dir: Directory where ``X.npy`` and bundle sidecars are written.
        sequence_col: Column containing transcript sequence strings.
        id_col: Column containing transcript or example identifiers.
        target_col: Optional column containing scalar regression targets.
        cds_positions_col: Optional column containing CDS position lists in
            transcript coordinates.
        splice_positions_col: Optional column containing splice position lists
            in transcript coordinates.
        length: Fixed Saluki input length to encode for every transcript.
        metadata_cols: Optional columns to copy into bundle metadata. When
            omitted, non-input and non-target columns are kept.
        split_col: Optional column with train/validation/test split labels.
        delimiter: Optional table delimiter override.
        progress: Whether to emit progress messages while building the bundle.
    """

    log_progress(f"build-saluki: reading {table_path}", enabled=progress)
    rows = _read_rows(table_path, delimiter=delimiter)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 6, int(length)))
    reporter = ProgressReporter(
        "build-saluki: encode transcripts",
        total=len(rows),
        unit="transcripts",
        enabled=progress,
    )
    for i, row in enumerate(rows):
        X[i] = encode_saluki_transcript(
            row[sequence_col],
            length=length,
            cds_positions=_parse_positions(row.get(cds_positions_col) if cds_positions_col else None),
            splice_positions=_parse_positions(row.get(splice_positions_col) if splice_positions_col else None),
        )
        reporter.update()
    X.flush()
    reporter.close()
    y = None
    if target_col is not None:
        log_progress("build-saluki: writing targets", enabled=progress)
        y = np.array([float(row[target_col]) for row in rows], dtype=np.float32)
        np.save(out / "y.npy", y)
    ids = [str(row[id_col]) for row in rows]
    exclude = {sequence_col, id_col}
    for col in (target_col, cds_positions_col, splice_positions_col):
        if col:
            exclude.add(col)
    metadata = [_metadata_for_row(row, exclude, metadata_cols) for row in rows]
    bundle = DatasetBundle(
        X=X,
        y=y,
        ids=ids,
        schema=SALUKI6,
        metadata=metadata,
        splits=_splits_from_rows(rows, split_col),
        config={
            "builder": "saluki_table",
            "source": str(table_path),
            "sequence_col": sequence_col,
            "target_col": target_col,
            "id_col": id_col,
            "split_col": split_col,
            "cds_positions_col": cds_positions_col,
            "splice_positions_col": splice_positions_col,
            "length": int(length),
        },
    )
    log_progress(f"build-saluki: saving metadata to {out}", enabled=progress)
    save_bundle_metadata(bundle, out)
    log_progress("build-saluki: done", enabled=progress)
    return bundle


def build_saluki_dataset_from_gtf(
    *,
    gtf_path: str | Path,
    fasta_path: str | Path,
    out_dir: str | Path,
    targets_path: str | Path | None = None,
    target_col: str | None = None,
    target_id_col: str = "transcript_id",
    length: int = DEFAULT_SALUKI_LENGTH,
    metadata_cols: Sequence[str] | None = None,
    split_col: str | None = None,
    delimiter: str | None = None,
    progress: bool = True,
) -> DatasetBundle:
    """Build a Saluki-style dataset directly from transcript GTF and genome FASTA.

    GTF parsing is implemented in pure Python to avoid pyranges/rtracklayer GTF
    compatibility issues. FASTA access uses ``pyfaidx`` when installed and falls
    back to a small in-memory reader for tests or tiny toy genomes.

    Args:
        gtf_path: GTF annotation file containing transcript exon and CDS
            features.
        fasta_path: Genome FASTA file used to assemble spliced transcript
            sequences.
        out_dir: Directory where ``X.npy`` and bundle sidecars are written.
        targets_path: Optional CSV/TSV-like target table used to select
            transcripts and provide labels or metadata.
        target_col: Optional target-table column containing scalar regression
            targets.
        target_id_col: Target-table column containing transcript identifiers
            that match GTF ``transcript_id`` attributes.
        length: Fixed Saluki input length to encode for every transcript.
        metadata_cols: Optional target-table columns to copy into bundle
            metadata.
        split_col: Optional target-table column with train/validation/test split
            labels.
        delimiter: Optional target-table delimiter override.
        progress: Whether to emit progress messages while building the bundle.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_rows: list[dict[str, str]] | None = None
    target_by_id: dict[str, Mapping[str, str]] | None = None
    transcript_ids: set[str] | None = None

    # Read table of values to predict (e.g., log(kdeg)'s)
    if targets_path is not None:
        log_progress(f"build-saluki-gtf: reading targets {targets_path}", enabled=progress)
        
        # target table as a Sequence of dicts, one per row
        target_rows = _read_rows(targets_path, delimiter=delimiter)

        # dict where key = transcript_id and value = other columns in target table
        target_by_id = _dedupe_target_rows(target_rows, target_id_col)

        # A bit cutesy: set() on dict returns set of keys
        transcript_ids = set(target_by_id)

    # Parse GTF
    features_by_id = load_transcript_features(gtf_path, transcript_ids=transcript_ids, progress=progress)
        # Dictionary with keys = transcript ID and values = TranscriptFeature class
        # TranscriptFeature class has attributes:
            # i) transcript_id
            # ii) chrom
            # iii) strand
            # iv) exons: tuple of GTF info (GTFRecord class) for each exon in transcript
            # v) cds: tuple of GTF info for each exonic CDS poritions in transcript

    if target_rows is None:
        selected_ids = list(features_by_id)
        selected_target_rows = None
    else:
        selected_ids = [str(row[target_id_col]) for row in target_rows if str(row[target_id_col]) in features_by_id]
        selected_target_rows = [target_by_id[tid] for tid in selected_ids] if target_by_id is not None else None
    if not selected_ids:
        raise ValueError("No transcripts with exon annotations were found for the requested inputs")
    n_missing_gtf_transcripts = len(target_rows) - len(selected_ids) if target_rows is not None else None

    # Make the sequence/CDS/splice-site tensor
    log_progress(f"build-saluki-gtf: opening FASTA {fasta_path}", enabled=progress)
    fasta = _FastaAccessor(fasta_path)
    try:
        kept_ids: list[str] = []
        kept_target_rows: list[Mapping[str, str]] | None = [] if selected_target_rows is not None else None
        skipped_fasta_chromosomes: dict[str, int] = {}
        reporter = ProgressReporter(
            "build-saluki-gtf: check FASTA chromosomes",
            total=len(selected_ids),
            unit="transcripts",
            enabled=progress,
        )

        # Filter for chromosomes in GTF
        for i, tid in enumerate(selected_ids):
            chrom = features_by_id[tid].chrom
            if not fasta.has_chrom(chrom):
                skipped_fasta_chromosomes[chrom] = skipped_fasta_chromosomes.get(chrom, 0) + 1
                reporter.update()
                continue
            kept_ids.append(tid)
            if kept_target_rows is not None and selected_target_rows is not None:
                kept_target_rows.append(selected_target_rows[i])
            reporter.update()
        reporter.close(extra=f"{len(kept_ids)} kept")
        if not kept_ids:
            raise ValueError("No transcripts remained after filtering to chromosomes present in the FASTA")

        # Create encoding tensor
            # Memmapped output 
        selected_ids = kept_ids
        selected_target_rows = kept_target_rows
        X = np.lib.format.open_memmap(
            out / "X.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(selected_ids), 6, int(length)),
        )
        ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        reporter = ProgressReporter(
            "build-saluki-gtf: encode transcripts",
            total=len(selected_ids),
            unit="transcripts",
            enabled=progress,
        )

        # Go transcript by transcript and generate its encoding
        for i, tid in enumerate(selected_ids):
            # This is the money function that does all the interesting heavy lifting
                # Builds an exon map from the TranscriptFeature object
                # Gets the transcript isoform RNA sequence
                # Identifies splice positions
                # Identifies codon positions
                # Gets any and all metadata
                # Stores it in TranscriptRecord object
            record = transcript_record_from_feature(features_by_id[tid], fasta)
            X[i] = encode_saluki_transcript(
                record.sequence,
                length=int(length),
                cds_positions=record.cds_positions,
                splice_positions=record.splice_positions,
            )
            row_meta = dict(record.metadata)
            if selected_target_rows is not None:
                target_row = selected_target_rows[i]
                exclude = {target_id_col}
                if target_col:
                    exclude.add(target_col)
                row_meta.update(_metadata_for_row(target_row, exclude, metadata_cols))
            ids.append(tid)
            metadata.append(row_meta)
            reporter.update()
        X.flush()
        reporter.close()
    finally:
        fasta.close()

    # Generate target tensor if relevant
    y = None
    if target_col is not None:
        if selected_target_rows is None:
            raise ValueError("target_col requires targets_path")
        log_progress("build-saluki-gtf: writing targets", enabled=progress)
        y = np.array([float(row[target_col]) for row in selected_target_rows], dtype=np.float32)
        np.save(out / "y.npy", y)
    splits = _splits_from_rows(selected_target_rows, split_col) if selected_target_rows is not None else None

    # Save metadata
    bundle = DatasetBundle(
        X=X,
        y=y,
        ids=ids,
        schema=SALUKI6,
        metadata=metadata,
        splits=splits,
        config={
            "builder": "saluki_gtf",
            "gtf": str(gtf_path),
            "fasta": str(fasta_path),
            "targets": str(targets_path) if targets_path is not None else None,
            "target_col": target_col,
            "target_id_col": target_id_col,
            "split_col": split_col,
            "length": int(length),
            "n_requested_targets": len(target_rows) if target_rows is not None else None,
            "n_missing_transcripts": n_missing_gtf_transcripts,
            "n_missing_gtf_transcripts": n_missing_gtf_transcripts,
            "n_skipped_missing_fasta_chromosome": int(sum(skipped_fasta_chromosomes.values())),
            "skipped_missing_fasta_chromosomes": dict(sorted(skipped_fasta_chromosomes.items())),
        },
    )
    log_progress(f"build-saluki-gtf: saving metadata to {out}", enabled=progress)
    save_bundle_metadata(bundle, out)
    log_progress("build-saluki-gtf: done", enabled=progress)
    return bundle
