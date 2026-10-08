#!/usr/bin/env python3
"""Build one Saluki-6 bundle from endogenous transcripts and MPRA reporters."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# Make the script runnable from a source checkout while retaining support for a
# copied script that uses an installed TranscriptML package or PYTHONPATH.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from transcriptml.data.bundle import DatasetBundle, save_bundle_metadata
from transcriptml.data.builders import build_saluki_dataset_from_gtf
from transcriptml.data.encoding import DEFAULT_SALUKI_LENGTH, encode_saluki_transcript
from transcriptml.data.schemas import SALUKI6
from transcriptml.progress import ProgressReporter, log_progress


CONSTRUCT_SEGMENTS = ("5utr", "orf", "3utr.1", "3utr.2")
SPLIT_NAMES = ("train", "val", "test")


def _csv_columns(value: str | None) -> list[str] | None:
    """Parse an optional comma-separated list of table columns."""

    if value is None:
        return None
    return [column.strip() for column in value.split(",") if column.strip()]


def _infer_delimiter(path: Path, delimiter: str | None) -> str:
    """Match TranscriptML's CSV/TSV delimiter inference."""

    if delimiter is not None:
        return delimiter
    return "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","


def _read_table(path: Path, delimiter: str | None) -> tuple[list[str], list[dict[str, str]]]:
    """Read a delimited table and retain its header even when it has no rows."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=_infer_delimiter(path, delimiter))
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _require_columns(path: Path, fieldnames: Sequence[str], columns: Sequence[str | None]) -> None:
    """Fail early when a requested input-table column is absent."""

    available = set(fieldnames)
    missing = sorted({column for column in columns if column and column not in available})
    if missing:
        raise ValueError(f"Missing column(s) in {path}: {', '.join(missing)}")


def _metadata_for_row(
    row: Mapping[str, str],
    *,
    exclude: set[str],
    metadata_cols: Sequence[str] | None,
) -> dict[str, str | None]:
    """Select metadata with the same default exclusions as TranscriptML builders."""

    columns = metadata_cols if metadata_cols is not None else [name for name in row if name not in exclude]
    return {name: row.get(name) for name in columns if name in row}


def _splits_from_rows(
    rows: Sequence[Mapping[str, str]], split_col: str | None
) -> dict[str, list[int]] | None:
    """Reproduce the existing builder's permissive predefined-split behavior."""

    if split_col is None:
        return None
    splits = {name: [] for name in SPLIT_NAMES}
    aliases = {"val": "val", "valid": "val", "validation": "val", "train": "train", "test": "test"}
    unknown: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = str(row.get(split_col, "")).strip().lower()
        if value in aliases:
            splits[aliases[value]].append(index)
        elif value:
            unknown[value] = unknown.get(value, 0) + 1
    if unknown:
        details = ", ".join(f"{label}={count}" for label, count in sorted(unknown.items()))
        raise ValueError(
            f"Unknown split label(s) in column '{split_col}': {details}. "
            "Expected train, val/valid/validation, or test."
        )
    return splits


def _read_construct(path: Path) -> dict[str, str]:
    """Read and validate the four labeled reporter-construct segments."""

    segments: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"Expected exactly two tab-separated fields at {path}:{line_number}"
                )
            label, sequence = (field.strip() for field in fields)
            if label not in CONSTRUCT_SEGMENTS:
                raise ValueError(
                    f"Unexpected construct segment {label!r} at {path}:{line_number}; "
                    f"expected {', '.join(CONSTRUCT_SEGMENTS)}"
                )
            if label in segments:
                raise ValueError(f"Duplicate construct segment {label!r} at {path}:{line_number}")
            if not sequence:
                raise ValueError(f"Construct segment {label!r} is empty at {path}:{line_number}")
            segments[label] = sequence

    missing = [label for label in CONSTRUCT_SEGMENTS if label not in segments]
    if missing:
        raise ValueError(f"Missing construct segment(s) in {path}: {', '.join(missing)}")
    if len(segments["orf"]) % 3:
        raise ValueError(
            f"Construct ORF length must be divisible by three; found {len(segments['orf'])} nt"
        )
    return segments


def _parse_targets(
    rows: Sequence[Mapping[str, str]], *, column: str, table: Path
) -> np.ndarray:
    """Convert a scalar target column to TranscriptML's float32 representation."""

    values: list[float] = []
    for index, row in enumerate(rows):
        try:
            values.append(float(row[column]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric target in {table}, row {index + 2}, column {column!r}: "
                f"{row.get(column)!r}"
            ) from exc
    return np.asarray(values, dtype=np.float32)


def _merge_splits(
    endogenous: Mapping[str, Sequence[int]] | None,
    mpra: Mapping[str, Sequence[int]] | None,
    *,
    mpra_offset: int,
) -> dict[str, list[int]] | None:
    """Merge optional source splits while preserving unassigned rows."""

    if endogenous is None and mpra is None:
        return None
    merged = {name: [] for name in SPLIT_NAMES}
    if endogenous is not None:
        for name in SPLIT_NAMES:
            merged[name].extend(int(index) for index in endogenous.get(name, ()))
    if mpra is not None:
        for name in SPLIT_NAMES:
            merged[name].extend(mpra_offset + int(index) for index in mpra.get(name, ()))
    return merged


def _ensure_output_available(path: Path) -> None:
    """Require an absent or empty output directory."""

    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {path}")
    if any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")


def _close_memmap(array: np.ndarray | None) -> None:
    """Close a NumPy memmap before removing its temporary backing file."""

    if isinstance(array, np.memmap) and getattr(array, "_mmap", None) is not None:
        if not array._mmap.closed:
            array._mmap.close()


def _publish(staged_bundle: Path, out_dir: Path) -> None:
    """Atomically publish a staged bundle on the destination filesystem."""

    _ensure_output_available(out_dir)
    if out_dir.exists():
        out_dir.rmdir()
    staged_bundle.replace(out_dir)


def build_fusion(args: argparse.Namespace) -> Path:
    """Build and publish the fused bundle described by parsed CLI arguments."""

    endogenous_table = Path(args.endogenous_table)
    mpra_table = Path(args.mpra_table)
    construct_path = Path(args.construct)
    out_dir = Path(args.out_dir)
    if int(args.length) <= 0:
        raise ValueError("length must be positive")
    _ensure_output_available(out_dir)

    endogenous_metadata_cols = _csv_columns(args.endogenous_metadata_cols)
    mpra_metadata_cols = _csv_columns(args.mpra_metadata_cols)

    endogenous_fields, _ = _read_table(endogenous_table, args.endogenous_delimiter)
    _require_columns(
        endogenous_table,
        endogenous_fields,
        [
            args.endogenous_id_col,
            args.endogenous_target_col,
            args.endogenous_split_col,
            *(endogenous_metadata_cols or ()),
        ],
    )
    mpra_fields, mpra_rows = _read_table(mpra_table, args.mpra_delimiter)
    _require_columns(
        mpra_table,
        mpra_fields,
        [
            args.mpra_sequence_col,
            args.mpra_target_col,
            args.mpra_id_col,
            args.mpra_split_col,
            *(mpra_metadata_cols or ()),
        ],
    )
    segments = _read_construct(construct_path)
    mpra_targets = _parse_targets(mpra_rows, column=args.mpra_target_col, table=mpra_table)
    mpra_splits = _splits_from_rows(mpra_rows, args.mpra_split_col)

    if args.mpra_id_col:
        mpra_source_ids = [str(row[args.mpra_id_col]) for row in mpra_rows]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for source_id in mpra_source_ids:
            if source_id in seen:
                duplicates.add(source_id)
            seen.add(source_id)
        if duplicates:
            example = sorted(duplicates)[0]
            raise ValueError(f"Duplicate MPRA ID in column {args.mpra_id_col!r}: {example!r}")
    else:
        mpra_source_ids = [str(index) for index in range(len(mpra_rows))]

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging.", dir=str(out_dir.parent))
    )
    endogenous_dir = staging_root / "endogenous"
    fused_dir = staging_root / "bundle"
    endogenous_bundle: DatasetBundle | None = None
    fused_x: np.memmap | None = None
    try:
        log_progress("build-fusion: build endogenous source")
        endogenous_bundle = build_saluki_dataset_from_gtf(
            gtf_path=args.gtf,
            fasta_path=args.fasta,
            out_dir=endogenous_dir,
            targets_path=endogenous_table,
            target_col=args.endogenous_target_col,
            target_id_col=args.endogenous_id_col,
            length=int(args.length),
            truncate_from=args.truncate_from,
            metadata_cols=endogenous_metadata_cols,
            split_col=args.endogenous_split_col,
            delimiter=args.endogenous_delimiter,
        )
        if endogenous_bundle.y is None:
            raise RuntimeError("Endogenous builder did not produce y.npy")

        n_endogenous = int(endogenous_bundle.X.shape[0])
        n_mpra = len(mpra_rows)
        total = n_endogenous + n_mpra
        log_progress(
            f"build-fusion: allocate fused array for {total} examples "
            f"({n_endogenous} endogenous, {n_mpra} MPRA)"
        )
        fused_dir.mkdir()
        fused_x = np.lib.format.open_memmap(
            fused_dir / "X.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, 6, int(args.length)),
        )
        copy_reporter = ProgressReporter(
            "build-fusion: copy endogenous encodings",
            total=n_endogenous,
            unit="examples",
        )
        for start in range(0, n_endogenous, 256):
            end = min(n_endogenous, start + 256)
            fused_x[start:end] = endogenous_bundle.X[start:end]
            copy_reporter.update(current=end)
        copy_reporter.close()

        five_utr = segments["5utr"]
        orf = segments["orf"]
        three_utr_1 = segments["3utr.1"]
        three_utr_2 = segments["3utr.2"]
        five_utr_start = 0
        five_utr_end = len(five_utr)
        cds_start = five_utr_end
        cds_end = cds_start + len(orf)
        three_utr_1_start = cds_end
        three_utr_1_end = three_utr_1_start + len(three_utr_1)
        cds_positions = range(cds_start, cds_end, 3)
        terminal_codon = orf[-3:].upper().replace("U", "T")

        endogenous_ids = [f"endogenous:{source_id}" for source_id in endogenous_bundle.ids]
        ids = endogenous_ids + [f"mpra:{source_id}" for source_id in mpra_source_ids]
        metadata: list[dict[str, object]] = []
        for source_id, row_metadata in zip(
            endogenous_bundle.ids, endogenous_bundle.metadata or [{} for _ in endogenous_bundle.ids]
        ):
            item = dict(row_metadata)
            item["data_source"] = "endogenous"
            item["source_id"] = str(source_id)
            metadata.append(item)

        mpra_exclude = {args.mpra_sequence_col, args.mpra_target_col}
        if args.mpra_id_col:
            mpra_exclude.add(args.mpra_id_col)
        mpra_reporter = ProgressReporter(
            "build-fusion: encode MPRA reporters",
            total=n_mpra,
            unit="reporters",
        )
        for mpra_index, (row, source_id) in enumerate(zip(mpra_rows, mpra_source_ids)):
            insert = str(row[args.mpra_sequence_col])
            insert_start = three_utr_1_end
            insert_end = insert_start + len(insert)
            three_utr_2_start = insert_end
            three_utr_2_end = three_utr_2_start + len(three_utr_2)
            sequence = five_utr + orf + three_utr_1 + insert + three_utr_2
            fused_x[n_endogenous + mpra_index] = encode_saluki_transcript(
                sequence,
                length=int(args.length),
                cds_positions=cds_positions,
                splice_positions=(),
                truncate_from=args.truncate_from,
            )
            transcript_length = len(sequence)
            encoded_offset = (
                max(0, transcript_length - int(args.length))
                if args.truncate_from == "5prime"
                else 0
            )
            item: dict[str, object] = _metadata_for_row(
                row,
                exclude=mpra_exclude,
                metadata_cols=mpra_metadata_cols,
            )
            item.update(
                {
                    "data_source": "mpra",
                    "source_id": source_id,
                    "transcript_length": transcript_length,
                    "represented_length": min(transcript_length, int(args.length)),
                    "encoded_offset": encoded_offset,
                    "exon_count": 1,
                    "cds_length": len(orf),
                    "five_prime_utr_start": five_utr_start,
                    "five_prime_utr_end": five_utr_end,
                    "cds_start": cds_start,
                    "cds_end": cds_end,
                    "three_prime_utr_1_start": three_utr_1_start,
                    "three_prime_utr_1_end": three_utr_1_end,
                    "insert_start": insert_start,
                    "insert_end": insert_end,
                    "three_prime_utr_2_start": three_utr_2_start,
                    "three_prime_utr_2_end": three_utr_2_end,
                }
            )
            metadata.append(item)
            mpra_reporter.update()
        mpra_reporter.close()

        fused_x.flush()
        log_progress("build-fusion: write targets and bundle metadata")
        y = np.concatenate(
            (np.asarray(endogenous_bundle.y, dtype=np.float32), mpra_targets)
        ).astype(np.float32, copy=False)
        np.save(fused_dir / "y.npy", y)
        splits = _merge_splits(
            endogenous_bundle.splits,
            mpra_splits,
            mpra_offset=n_endogenous,
        )
        endogenous_filtering = {
            key: endogenous_bundle.config.get(key)
            for key in (
                "n_requested_targets",
                "n_missing_transcripts",
                "n_missing_gtf_transcripts",
                "n_skipped_missing_fasta_chromosome",
                "skipped_missing_fasta_chromosomes",
            )
        }
        config = {
            "builder": "mpra_endogenous_fusion",
            "length": int(args.length),
            "truncate_from": args.truncate_from,
            "n_endogenous": n_endogenous,
            "n_mpra": n_mpra,
            "example_order": ["endogenous", "mpra"],
            "target_scale_assumption": "already_comparable_and_concatenated_unchanged",
            "endogenous": {
                "table": str(endogenous_table),
                "gtf": str(args.gtf),
                "fasta": str(args.fasta),
                "id_col": args.endogenous_id_col,
                "target_col": args.endogenous_target_col,
                "split_col": args.endogenous_split_col,
                "metadata_cols": endogenous_metadata_cols,
                "delimiter": args.endogenous_delimiter,
                "filtering": endogenous_filtering,
            },
            "mpra": {
                "table": str(mpra_table),
                "sequence_col": args.mpra_sequence_col,
                "id_col": args.mpra_id_col,
                "target_col": args.mpra_target_col,
                "split_col": args.mpra_split_col,
                "metadata_cols": mpra_metadata_cols,
                "delimiter": args.mpra_delimiter,
                "insert_orientation": "as_written_5prime_to_3prime",
            },
            "construct": {
                "source": str(construct_path),
                "segment_order": ["5utr", "orf", "3utr.1", "insert", "3utr.2"],
                "segment_lengths": {label: len(segments[label]) for label in CONSTRUCT_SEGMENTS},
                "terminal_codon": terminal_codon,
                "orf_includes_terminal_stop": terminal_codon in {"TAA", "TAG", "TGA"},
                "splice_junctions": 0,
            },
        }
        sidecar_bundle = DatasetBundle(
            X=fused_x,
            y=y,
            ids=ids,
            schema=SALUKI6,
            metadata=metadata,
            splits=splits,
            config=config,
        )
        save_bundle_metadata(sidecar_bundle, fused_dir)
        _close_memmap(fused_x)
        fused_x = None
        _close_memmap(endogenous_bundle.X)
        log_progress(f"build-fusion: publish bundle to {out_dir}")
        _publish(fused_dir, out_dir)
        log_progress(
            f"build-fusion: done ({n_endogenous} endogenous + {n_mpra} MPRA = {total} examples)"
        )
        return out_dir
    finally:
        _close_memmap(fused_x)
        if endogenous_bundle is not None:
            _close_memmap(endogenous_bundle.X)
        shutil.rmtree(staging_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone fusion CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build a fused endogenous + MPRA Saluki-6 TranscriptML bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--endogenous-table", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--endogenous-target-col", required=True)
    parser.add_argument("--endogenous-id-col", default="transcript_id")
    parser.add_argument("--endogenous-metadata-cols")
    parser.add_argument("--endogenous-split-col")
    parser.add_argument("--endogenous-delimiter")
    parser.add_argument("--mpra-table", required=True)
    parser.add_argument("--mpra-sequence-col", required=True)
    parser.add_argument("--mpra-target-col", required=True)
    parser.add_argument("--mpra-id-col")
    parser.add_argument("--mpra-metadata-cols")
    parser.add_argument("--mpra-split-col")
    parser.add_argument("--mpra-delimiter")
    parser.add_argument("--construct", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--length", type=int, default=DEFAULT_SALUKI_LENGTH)
    parser.add_argument("--truncate-from", choices=("5prime", "3prime"), default="5prime")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone CLI."""

    args = build_parser().parse_args(argv)
    output = build_fusion(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
