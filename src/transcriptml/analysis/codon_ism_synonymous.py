#!/usr/bin/env python3
"""Summarize and plot synonymous codon ISM outputs across Saluki folds.

The codon_ism.py writer stores cds_relative_position as

    (cds_end - codon_start) / cds_length

which runs from high values at the 5' CDS end to low values at the 3' CDS end.
By default this script converts it to a plotting coordinate

    cds_position_5p_to_3p = 1 - cds_relative_position

before binning, so positional plots run from 5' to 3' on the x-axis.
Use --keep-input-position to plot the stored coordinate directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-codon-ism"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import polars as pl
import seaborn as sns


IDENTITY_COLUMNS = [
    "sequence_index",
    "codon_start",
    "cds_codon_index",
    "cds_start",
    "cds_end",
    "cds_length",
    "cds_relative_position",
    "reference_codon",
    "alternate_codon",
    "reference_amino_acid",
    "alternate_amino_acid",
    "synonymous",
]

PREDICTION_COLUMNS = ["reference_prediction", "mutant_prediction", "delta"]
REQUIRED_COLUMNS = IDENTITY_COLUMNS + PREDICTION_COLUMNS
POSITION_COLUMN = "cds_position_5p_to_3p"

# RNA codons in the standard genetic code. The family order follows the common
# codon-table row layout so tables and plots are stable across runs.
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
CODON_SORT = {
    codon: i for i, codon in enumerate(codon for codons in CODONS_BY_AA.values() for codon in codons)
}


@dataclass(frozen=True)
class DatasetOutputs:
    """Small tables used by downstream plotting and combined summaries."""

    label: str
    global_table: pl.DataFrame
    aa_position_table: pl.DataFrame
    codon_position_table: pl.DataFrame


def natural_fold_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"fold(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def default_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    base_dir = default_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Summarize synonymous codon ISM parquet files, average mutations "
            "across folds, and create global and positional plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fold-dirs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Fold directories containing mutations.parquet. If omitted, uses "
            "<script-parent>/codon_ism/fold[0-9]."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "codon_ism_synonymous_summary",
        help="Directory for tables, plots, README, and average_mutations.parquet.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of equal-width relative CDS position bins.",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help=(
            "Use -delta as the plotted/ranked effect. By default effect equals "
            "delta = mutant_prediction - reference_prediction."
        ),
    )
    parser.add_argument(
        "--keep-input-position",
        action="store_true",
        help=(
            "Do not convert cds_relative_position to 5'-to-3' coordinates before "
            "binning. The default plots 1 - cds_relative_position."
        ),
    )
    parser.add_argument(
        "--missing-folds",
        choices=("warn", "fail", "ignore"),
        default="warn",
        help=(
            "How to handle averaged mutation keys whose n_folds is less than the "
            "number of input fold directories."
        ),
    )
    parser.add_argument(
        "--plots",
        choices=("all", "average-only", "folds-only", "none"),
        default="all",
        help="Which positional plot sets to render.",
    )
    parser.add_argument(
        "--reuse-average",
        action="store_true",
        help=(
            "Reuse an existing <out-dir>/average_mutations.parquet instead of "
            "recomputing it."
        ),
    )
    parser.add_argument(
        "--average-method",
        choices=("auto", "row-order", "group-by"),
        default="auto",
        help=(
            "Method for building average_mutations.parquet. 'row-order' verifies "
            "that identity columns match batch-by-batch across folds, then averages "
            "without a large group-by. 'group-by' aligns strictly by mutation keys. "
            "'auto' tries row-order first and falls back to group-by if alignment fails."
        ),
    )
    parser.add_argument(
        "--average-batch-size",
        type=int,
        default=65_536,
        help="Rows per batch for row-order averaging.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG plot resolution.",
    )
    parser.add_argument(
        "--limit-rows-per-fold",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.n_bins <= 0:
        parser.error("--n-bins must be positive")
    if args.limit_rows_per_fold is not None and args.limit_rows_per_fold <= 0:
        parser.error("--limit-rows-per-fold must be positive when provided")
    if args.average_batch_size <= 0:
        parser.error("--average-batch-size must be positive")
    return args


def resolve_fold_dirs(fold_dirs: list[Path] | None) -> list[Path]:
    if fold_dirs:
        resolved = [path.resolve() for path in fold_dirs]
    else:
        parent = default_base_dir() / "codon_ism"
        resolved = sorted(parent.glob("fold[0-9]"), key=natural_fold_key)

    if not resolved:
        raise SystemExit("No fold directories were found or provided.")

    missing = [str(path / "mutations.parquet") for path in resolved if not (path / "mutations.parquet").exists()]
    if missing:
        raise SystemExit("Missing mutations.parquet files:\n" + "\n".join(missing))

    return sorted(resolved, key=natural_fold_key)


def validate_schema(parquet_path: Path) -> None:
    schema = pl.scan_parquet(parquet_path).collect_schema()
    missing = [column for column in REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise SystemExit(f"{parquet_path} is missing required columns: {', '.join(missing)}")


def scan_mutations(parquet_path: Path, limit_rows: int | None = None) -> pl.LazyFrame:
    validate_schema(parquet_path)
    lazy = pl.scan_parquet(parquet_path).select(REQUIRED_COLUMNS)
    if limit_rows is not None:
        lazy = lazy.head(limit_rows)
    return lazy


def effect_expr(flip_sign: bool) -> pl.Expr:
    expr = -pl.col("delta") if flip_sign else pl.col("delta")
    return expr.cast(pl.Float64).alias("effect")


def add_effect_and_position(
    lazy: pl.LazyFrame,
    *,
    flip_sign: bool,
    keep_input_position: bool,
    n_bins: int,
) -> pl.LazyFrame:
    input_position = pl.col("cds_relative_position").cast(pl.Float64)
    position = input_position if keep_input_position else (1.0 - input_position)
    raw_bin = (pl.col(POSITION_COLUMN) * n_bins).floor().cast(pl.Int64)
    return (
        lazy.with_columns(
            [
                effect_expr(flip_sign),
                position.clip(0.0, 1.0).alias(POSITION_COLUMN),
            ]
        )
        .with_columns(
            [
                pl.when(raw_bin < 0)
                .then(0)
                .when(raw_bin >= n_bins)
                .then(n_bins - 1)
                .otherwise(raw_bin)
                .alias("position_bin")
            ]
        )
    )


def add_bin_columns(data: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    return data.with_columns(
        [
            (pl.col("position_bin").cast(pl.Float64) / n_bins).alias("bin_start"),
            ((pl.col("position_bin").cast(pl.Float64) + 1.0) / n_bins).alias("bin_end"),
            ((pl.col("position_bin").cast(pl.Float64) + 0.5) / n_bins).alias("bin_center"),
        ]
    )


def with_order_columns(data: pl.DataFrame) -> pl.DataFrame:
    aa_values = data.get_column("reference_amino_acid").to_list()
    codon_values = data.get_column("reference_codon").to_list()
    columns = [
        pl.Series("amino_acid_order", [AA_SORT.get(value, 10**9) for value in aa_values], dtype=pl.Int64),
        pl.Series("codon_order", [CODON_SORT.get(value, 10**9) for value in codon_values], dtype=pl.Int64),
    ]
    if "alternate_codon" in data.columns:
        alt_values = data.get_column("alternate_codon").to_list()
        columns.append(
            pl.Series("alternate_codon_order", [CODON_SORT.get(value, 10**9) for value in alt_values], dtype=pl.Int64)
        )
    return data.with_columns(columns)


def drop_order_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.drop([column for column in data.columns if column.endswith("_order")])


def add_amino_acid_names(data: pl.DataFrame) -> pl.DataFrame:
    aa_values = data.get_column("reference_amino_acid").to_list()
    return data.with_columns(
        pl.Series("amino_acid_name", [AA_NAMES.get(value, value) for value in aa_values], dtype=pl.String)
    )


def rank_and_order_global(data: pl.DataFrame) -> pl.DataFrame:
    data = with_order_columns(add_amino_acid_names(data))
    data = data.with_columns(
        pl.col("mean_effect")
        .rank(method="ordinal", descending=True)
        .over("reference_amino_acid")
        .cast(pl.Int64)
        .alias("within_amino_acid_rank")
    )
    data = data.sort(["amino_acid_order", "within_amino_acid_rank", "codon_order"])
    ordered_columns = [
        "reference_amino_acid",
        "amino_acid_name",
        "reference_codon",
        "mean_delta",
        "mean_effect",
        "effect_std",
        "effect_sem",
        "n_mutation_rows",
        "n_unique_sequence_codon_instances",
        "n_synonymous_alternate_codons",
        "within_amino_acid_rank",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def write_table(data: pl.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(stem.with_suffix(".csv"))
    data.write_csv(stem.with_suffix(".tsv"), separator="\t")


def summarize_global_table(
    parquet_path: Path,
    *,
    flip_sign: bool,
    limit_rows: int | None,
) -> pl.DataFrame:
    lazy = scan_mutations(parquet_path, limit_rows).with_columns(effect_expr(flip_sign))
    summary = (
        lazy.group_by(["reference_amino_acid", "reference_codon"])
        .agg(
            [
                pl.col("delta").cast(pl.Float64).mean().alias("mean_delta"),
                pl.col("effect").mean().alias("mean_effect"),
                pl.col("effect").std().alias("effect_std"),
                (pl.col("effect").std() / pl.len().cast(pl.Float64).sqrt()).alias("effect_sem"),
                pl.len().alias("n_mutation_rows"),
                pl.struct(["sequence_index", "codon_start", "cds_codon_index"])
                .n_unique()
                .alias("n_unique_sequence_codon_instances"),
                pl.col("alternate_codon").n_unique().alias("n_synonymous_alternate_codons"),
            ]
        )
        .collect()
    )
    return rank_and_order_global(summary)


def summarize_position_tables(
    parquet_path: Path,
    *,
    flip_sign: bool,
    keep_input_position: bool,
    n_bins: int,
    limit_rows: int | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = scan_mutations(parquet_path, limit_rows).select(
        [
            "sequence_index",
            "codon_start",
            "cds_codon_index",
            "cds_relative_position",
            "reference_amino_acid",
            "reference_codon",
            "alternate_codon",
            "delta",
        ]
    )
    positioned = add_effect_and_position(
        base,
        flip_sign=flip_sign,
        keep_input_position=keep_input_position,
        n_bins=n_bins,
    )
    shared_aggs = [
        pl.col("delta").cast(pl.Float64).mean().alias("mean_delta"),
        pl.col("effect").mean().alias("mean_effect"),
        pl.col("effect").std().alias("effect_std"),
        (pl.col("effect").std() / pl.len().cast(pl.Float64).sqrt()).alias("effect_sem"),
        pl.len().alias("n_mutation_rows"),
        pl.struct(["sequence_index", "codon_start", "cds_codon_index"])
        .n_unique()
        .alias("n_unique_sequence_codon_instances"),
    ]

    aa_bins = (
        positioned.group_by(["reference_amino_acid", "reference_codon", "position_bin"])
        .agg(shared_aggs)
        .collect()
    )
    aa_bins = add_bin_columns(with_order_columns(add_amino_acid_names(aa_bins)), n_bins)
    aa_bins = aa_bins.sort(["amino_acid_order", "codon_order", "position_bin"])
    aa_bins = drop_order_columns(aa_bins)

    codon_bins = (
        positioned.group_by(
            ["reference_amino_acid", "reference_codon", "alternate_codon", "position_bin"]
        )
        .agg(shared_aggs)
        .collect()
    )
    codon_bins = add_bin_columns(with_order_columns(add_amino_acid_names(codon_bins)), n_bins)
    codon_bins = codon_bins.sort(
        ["amino_acid_order", "codon_order", "alternate_codon_order", "position_bin"]
    )
    codon_bins = drop_order_columns(codon_bins)

    return aa_bins, codon_bins


def summarize_dataset(
    parquet_path: Path,
    *,
    label: str,
    table_dir: Path,
    flip_sign: bool,
    keep_input_position: bool,
    n_bins: int,
    limit_rows: int | None,
) -> DatasetOutputs:
    print(f"[{label}] Summarizing global codon effects from {parquet_path}", flush=True)
    global_table = summarize_global_table(parquet_path, flip_sign=flip_sign, limit_rows=limit_rows)
    write_table(global_table, table_dir / "global_codon_effects")

    print(f"[{label}] Summarizing positional bins", flush=True)
    aa_position, codon_position = summarize_position_tables(
        parquet_path,
        flip_sign=flip_sign,
        keep_input_position=keep_input_position,
        n_bins=n_bins,
        limit_rows=limit_rows,
    )
    write_table(aa_position, table_dir / "position_bins_by_amino_acid")
    write_table(codon_position, table_dir / "position_bins_by_reference_codon")
    return DatasetOutputs(
        label=label,
        global_table=global_table,
        aa_position_table=aa_position,
        codon_position_table=codon_position,
    )


def average_mutations(
    fold_dirs: list[Path],
    *,
    out_path: Path,
    limit_rows: int | None,
    method: str,
    batch_size: int,
) -> None:
    if method in {"auto", "row-order"}:
        try:
            average_mutations_row_order(
                fold_dirs,
                out_path=out_path,
                limit_rows=limit_rows,
                batch_size=batch_size,
            )
            return
        except Exception as exc:
            if method == "row-order":
                raise
            warnings.warn(
                f"Row-order averaging failed ({exc}); falling back to key-based group-by averaging."
            )

    average_mutations_group_by(fold_dirs, out_path=out_path, limit_rows=limit_rows)


def average_mutations_group_by(
    fold_dirs: list[Path],
    *,
    out_path: Path,
    limit_rows: int | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scans = []
    for fold_index, fold_dir in enumerate(fold_dirs):
        parquet_path = fold_dir / "mutations.parquet"
        lazy = scan_mutations(parquet_path, limit_rows)
        scans.append(lazy.with_columns(pl.lit(fold_dir.name).alias("fold_id"), pl.lit(fold_index).alias("fold_index")))

    combined = pl.concat(scans, how="vertical_relaxed")
    averaged = combined.group_by(IDENTITY_COLUMNS, maintain_order=False).agg(
        [
            pl.col("reference_prediction").cast(pl.Float64).mean().alias("reference_prediction"),
            pl.col("mutant_prediction").cast(pl.Float64).mean().alias("mutant_prediction"),
            pl.col("delta").cast(pl.Float64).mean().alias("delta"),
            pl.col("reference_prediction").cast(pl.Float64).std().alias("reference_prediction_std"),
            pl.col("mutant_prediction").cast(pl.Float64).std().alias("mutant_prediction_std"),
            pl.col("delta").cast(pl.Float64).std().alias("delta_std"),
            pl.col("fold_id").n_unique().alias("n_folds"),
            pl.len().alias("n_rows"),
        ]
    )

    print(f"[average] Writing key-aligned group-by fold average to {out_path}", flush=True)
    averaged.sink_parquet(out_path, compression="zstd", maintain_order=False, mkdir=True)


def average_mutations_row_order(
    fold_dirs: list[Path],
    *,
    out_path: Path,
    limit_rows: int | None,
    batch_size: int,
) -> None:
    """Average folds by row position after verifying mutation identity columns.

    codon_ism.py emits deterministic long-form rows for each fold in this
    project. When fold files are row-aligned, streaming through matching batches
    avoids materializing a 10-fold concatenation and grouping by tens of millions
    of mutation keys. Identity columns are still checked exactly for every batch.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_paths = [fold_dir / "mutations.parquet" for fold_dir in fold_dirs]
    for path in parquet_paths:
        validate_schema(path)

    parquet_files = [pq.ParquetFile(path) for path in parquet_paths]
    metadata_rows = [parquet_file.metadata.num_rows for parquet_file in parquet_files]
    expected_rows = min(metadata_rows) if limit_rows is None else min(min(metadata_rows), limit_rows)
    if limit_rows is None and len(set(metadata_rows)) != 1:
        raise ValueError(f"fold parquet row counts differ: {metadata_rows}")
    if limit_rows is not None and any(row_count < limit_rows for row_count in metadata_rows):
        raise ValueError(
            f"--limit-rows-per-fold={limit_rows} exceeds at least one fold row count: {metadata_rows}"
        )

    print(
        f"[average] Writing row-order verified fold average to {out_path} "
        f"({expected_rows} rows; batch_size={batch_size})",
        flush=True,
    )

    iterators = [
        parquet_file.iter_batches(batch_size=batch_size, columns=REQUIRED_COLUMNS)
        for parquet_file in parquet_files
    ]
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    batch_index = 0
    try:
        while total_rows < expected_rows:
            remaining = expected_rows - total_rows
            batches = []
            for iterator in iterators:
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"fold ended early after {total_rows} rows; expected {expected_rows}"
                    ) from exc
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                batches.append(batch)

            row_counts = [batch.num_rows for batch in batches]
            if len(set(row_counts)) != 1:
                raise ValueError(f"batch {batch_index} row counts differ across folds: {row_counts}")

            verify_batch_identity(batches, batch_index=batch_index)
            table = build_average_batch_table(batches)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += table.num_rows
            batch_index += 1
    finally:
        if writer is not None:
            writer.close()

    if total_rows != expected_rows:
        raise ValueError(f"wrote {total_rows} averaged rows, expected {expected_rows}")


def verify_batch_identity(batches: list[pa.RecordBatch], *, batch_index: int) -> None:
    first = batches[0]
    for fold_offset, batch in enumerate(batches[1:], start=1):
        for column in IDENTITY_COLUMNS:
            first_array = first.column(first.schema.get_field_index(column))
            other_array = batch.column(batch.schema.get_field_index(column))
            if not first_array.equals(other_array):
                raise ValueError(
                    f"identity mismatch in batch {batch_index}, fold offset {fold_offset}, "
                    f"column {column}"
                )


def build_average_batch_table(batches: list[pa.RecordBatch]) -> pa.Table:
    first = batches[0]
    arrays = [first.column(first.schema.get_field_index(column)) for column in IDENTITY_COLUMNS]
    names = list(IDENTITY_COLUMNS)
    n_folds = len(batches)
    n_rows = first.num_rows

    numeric_stacks: dict[str, np.ndarray] = {}
    for column in PREDICTION_COLUMNS:
        column_index = first.schema.get_field_index(column)
        numeric_stacks[column] = np.vstack(
            [
                np.asarray(batch.column(column_index).to_numpy(zero_copy_only=False), dtype=np.float64)
                for batch in batches
            ]
        )

    for column in PREDICTION_COLUMNS:
        values = numeric_stacks[column]
        arrays.append(pa.array(values.mean(axis=0), type=pa.float64()))
        names.append(column)

    std_columns = {
        "reference_prediction": "reference_prediction_std",
        "mutant_prediction": "mutant_prediction_std",
        "delta": "delta_std",
    }
    for column, output_column in std_columns.items():
        values = numeric_stacks[column]
        if n_folds > 1:
            std = values.std(axis=0, ddof=1)
        else:
            std = np.full(n_rows, np.nan, dtype=np.float64)
        arrays.append(pa.array(std, type=pa.float64()))
        names.append(output_column)

    arrays.append(pa.array(np.full(n_rows, n_folds, dtype=np.uint16), type=pa.uint16()))
    names.append("n_folds")
    arrays.append(pa.array(np.full(n_rows, n_folds, dtype=np.uint16), type=pa.uint16()))
    names.append("n_rows")
    return pa.Table.from_arrays(arrays, names=names)


def verify_average_mutations(
    average_path: Path,
    *,
    expected_folds: int,
    out_dir: Path,
    missing_folds: str,
) -> dict[str, int | float | None]:
    lazy = pl.scan_parquet(average_path)
    stats = lazy.select(
        [
            pl.len().alias("n_average_rows"),
            pl.col("n_folds").min().alias("min_n_folds"),
            pl.col("n_folds").max().alias("max_n_folds"),
            (pl.col("n_folds") != expected_folds).sum().alias("n_keys_not_in_all_folds"),
            (pl.col("n_rows") != pl.col("n_folds")).sum().alias("n_keys_with_duplicate_fold_rows"),
        ]
    ).collect()
    result = stats.to_dicts()[0]
    fold_counts = lazy.group_by("n_folds").agg(pl.len().alias("n_mutation_keys")).sort("n_folds").collect()
    write_table(fold_counts, out_dir / "average" / "fold_contribution_counts")

    incomplete = int(result["n_keys_not_in_all_folds"] or 0)
    duplicates = int(result["n_keys_with_duplicate_fold_rows"] or 0)
    message = (
        f"Average mutation integrity: {result['n_average_rows']} keys; "
        f"n_folds range {result['min_n_folds']}..{result['max_n_folds']}; "
        f"{incomplete} keys not present in all {expected_folds} folds; "
        f"{duplicates} keys with duplicate rows within a fold."
    )
    if incomplete or duplicates:
        if missing_folds == "fail":
            raise SystemExit(message)
        if missing_folds == "warn":
            warnings.warn(message)
        else:
            print(message, flush=True)
    else:
        print(message, flush=True)
    return result


def safe_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "unnamed"


def effect_axis_label(flip_sign: bool) -> str:
    if flip_sign:
        return "Mean effect (-delta)"
    return "Mean delta (mutant - reference)"


def sign_summary(flip_sign: bool) -> str:
    if flip_sign:
        return "Effect sign: effect = -delta; larger values are treated as more stabilizing."
    return (
        "Effect sign: effect = delta = mutant_prediction - reference_prediction; "
        "larger values are treated as more stabilizing."
    )


def position_axis_label(keep_input_position: bool) -> str:
    if keep_input_position:
        return "Relative CDS position (input cds_relative_position)"
    return "Relative CDS position (5' to 3')"


def setup_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "0.2",
            "grid.color": "0.88",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def codons_for_aa(aa: str, observed: list[str]) -> list[str]:
    preferred = [codon for codon in CODONS_BY_AA.get(aa, ()) if codon in observed]
    extra = sorted([codon for codon in observed if codon not in preferred])
    return preferred + extra


def warn_missing_reference_codons(aa: str, observed: list[str], label: str) -> None:
    expected = set(CODONS_BY_AA.get(aa, ()))
    missing = sorted(expected - set(observed), key=lambda codon: CODON_SORT.get(codon, 10**9))
    if missing:
        warnings.warn(
            f"[{label}] {AA_NAMES.get(aa, aa)} ({aa}) plot is missing expected reference "
            f"codon(s): {', '.join(missing)}. This can happen in small --limit-rows-per-fold "
            "debug runs or when those reference codons are absent from the dataset."
        )


def draw_lineplot(
    data,
    *,
    hue: str,
    hue_order: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str,
) -> plt.Figure:
    width = 6.8
    height = 4.2 if len(hue_order) <= 6 else 4.8
    fig, ax = plt.subplots(figsize=(width, height))
    palette = sns.color_palette("tab10", n_colors=max(len(hue_order), 3))
    sns.lineplot(
        data=data,
        x="bin_center",
        y="mean_effect",
        hue=hue,
        hue_order=hue_order,
        palette=palette[: len(hue_order)],
        marker="o",
        markersize=4.2,
        linewidth=1.5,
        ax=ax,
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    legend = ax.legend(title=legend_title, frameon=False, loc="best")
    if legend is not None:
        legend._legend_box.align = "left"
    fig.tight_layout()
    return fig


def plot_by_amino_acid(
    aa_position: pl.DataFrame,
    *,
    out_dir: Path,
    label: str,
    flip_sign: bool,
    keep_input_position: bool,
    dpi: int,
) -> int:
    setup_plot_style()
    n_plots = 0
    y_label = effect_axis_label(flip_sign)
    x_label = position_axis_label(keep_input_position)
    for aa in AA_ORDER:
        subset = aa_position.filter(pl.col("reference_amino_acid") == aa)
        if subset.is_empty():
            continue
        observed = subset.get_column("reference_codon").unique().to_list()
        warn_missing_reference_codons(aa, observed, label)
        hue_order = codons_for_aa(aa, observed)
        if not hue_order:
            continue
        pdf = subset.to_pandas()
        title = f"{AA_NAMES.get(aa, aa)} ({aa}) reference codons - {label}"
        fig = draw_lineplot(
            pdf,
            hue="reference_codon",
            hue_order=hue_order,
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            legend_title="Reference codon",
        )
        stem = out_dir / f"{safe_filename(aa)}_{safe_filename(AA_NAMES.get(aa, aa))}"
        save_figure(fig, stem, dpi)
        n_plots += 1
    return n_plots


def plot_by_reference_codon(
    codon_position: pl.DataFrame,
    *,
    out_dir: Path,
    label: str,
    flip_sign: bool,
    keep_input_position: bool,
    dpi: int,
) -> int:
    setup_plot_style()
    n_plots = 0
    y_label = effect_axis_label(flip_sign)
    x_label = position_axis_label(keep_input_position)
    ordered_pairs = []
    for aa in AA_ORDER:
        for codon in CODONS_BY_AA[aa]:
            ordered_pairs.append((aa, codon))

    for aa, codon in ordered_pairs:
        subset = codon_position.filter(
            (pl.col("reference_amino_acid") == aa) & (pl.col("reference_codon") == codon)
        )
        if subset.is_empty():
            continue
        observed_alts = subset.get_column("alternate_codon").unique().to_list()
        family_codons = [alt for alt in CODONS_BY_AA.get(aa, ()) if alt != codon]
        hue_order = [alt for alt in family_codons if alt in observed_alts]
        hue_order += sorted([alt for alt in observed_alts if alt not in hue_order])
        if not hue_order:
            continue
        pdf = subset.to_pandas()
        title = f"{AA_NAMES.get(aa, aa)} ({aa}) {codon} synonymous alternates - {label}"
        fig = draw_lineplot(
            pdf,
            hue="alternate_codon",
            hue_order=hue_order,
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            legend_title="Alternate codon",
        )
        stem = out_dir / f"{safe_filename(aa)}_{safe_filename(codon)}"
        save_figure(fig, stem, dpi)
        n_plots += 1
    return n_plots


def plot_dataset(
    outputs: DatasetOutputs,
    *,
    plot_root: Path,
    flip_sign: bool,
    keep_input_position: bool,
    dpi: int,
) -> tuple[int, int]:
    print(f"[{outputs.label}] Rendering positional plots", flush=True)
    aa_count = plot_by_amino_acid(
        outputs.aa_position_table,
        out_dir=plot_root / "by_amino_acid",
        label=outputs.label,
        flip_sign=flip_sign,
        keep_input_position=keep_input_position,
        dpi=dpi,
    )
    codon_count = plot_by_reference_codon(
        outputs.codon_position_table,
        out_dir=plot_root / "by_codon",
        label=outputs.label,
        flip_sign=flip_sign,
        keep_input_position=keep_input_position,
        dpi=dpi,
    )
    return aa_count, codon_count


def write_combined_global_table(outputs: list[DatasetOutputs], out_dir: Path) -> None:
    frames = []
    for item in outputs:
        frames.append(item.global_table.with_columns(pl.lit(item.label).alias("dataset")))
    if not frames:
        return
    combined = pl.concat(frames, how="vertical_relaxed")
    columns = ["dataset"] + [column for column in combined.columns if column != "dataset"]
    write_table(combined.select(columns), out_dir / "tables" / "global_codon_effects_by_dataset")


def write_readme(
    out_dir: Path,
    *,
    fold_dirs: list[Path],
    average_stats: dict[str, int | float | None],
    n_bins: int,
    flip_sign: bool,
    keep_input_position: bool,
    plots_mode: str,
    dataset_plot_counts: dict[str, tuple[int, int]],
    limit_rows: int | None,
    average_method: str,
) -> None:
    sign_line = (
        "Effect sign: effect = -delta; larger values are treated as more stabilizing."
        if flip_sign
        else "Effect sign: effect = delta = mutant_prediction - reference_prediction; larger values are treated as more stabilizing."
    )
    position_line = (
        "Position coordinate: used input cds_relative_position directly."
        if keep_input_position
        else "Position coordinate: converted to cds_position_5p_to_3p = 1 - cds_relative_position before binning."
    )
    plot_lines = [
        f"- {label}: {counts[0]} amino-acid plots, {counts[1]} reference-codon plots"
        for label, counts in sorted(dataset_plot_counts.items())
    ]
    if not plot_lines:
        plot_lines = ["- positional plot rendering was skipped"]
    folds = "\n".join(f"- {path}" for path in fold_dirs)
    stats_json = json.dumps(average_stats, indent=2, sort_keys=True)
    limit_line = f"\nDebug row limit per fold: {limit_rows}\n" if limit_rows is not None else ""
    text = f"""# Synonymous codon ISM summary

{sign_line}

{position_line}

Number of position bins: {n_bins}
Plots mode: {plots_mode}
Requested average method: {average_method}
{limit_line}
Input folds:
{folds}

Average mutation parquet:
- {out_dir / "average_mutations.parquet"}

Tables:
- {out_dir / "tables"}

Plots:
- {out_dir / "plots"}

Average mutation integrity:
```json
{stats_json}
```

Rendered plots:
{chr(10).join(plot_lines)}
"""
    (out_dir / "README.md").write_text(text)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    fold_dirs = resolve_fold_dirs(args.fold_dirs)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Synonymous codon ISM summarization", flush=True)
    print(f"Input folds: {', '.join(path.name for path in fold_dirs)}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(sign_summary(args.flip_sign), flush=True)
    print(position_axis_label(args.keep_input_position), flush=True)

    average_path = out_dir / "average_mutations.parquet"
    if average_path.exists() and args.reuse_average:
        print(f"[average] Reusing existing {average_path}", flush=True)
    else:
        average_mutations(
            fold_dirs,
            out_path=average_path,
            limit_rows=args.limit_rows_per_fold,
            method=args.average_method,
            batch_size=args.average_batch_size,
        )

    average_stats = verify_average_mutations(
        average_path,
        expected_folds=len(fold_dirs),
        out_dir=out_dir,
        missing_folds=args.missing_folds,
    )

    outputs: list[DatasetOutputs] = []
    for fold_dir in fold_dirs:
        label = fold_dir.name
        outputs.append(
            summarize_dataset(
                fold_dir / "mutations.parquet",
                label=label,
                table_dir=out_dir / "tables" / "per_fold" / label,
                flip_sign=args.flip_sign,
                keep_input_position=args.keep_input_position,
                n_bins=args.n_bins,
                limit_rows=args.limit_rows_per_fold,
            )
        )

    outputs.append(
        summarize_dataset(
            average_path,
            label="average",
            table_dir=out_dir / "tables" / "average",
            flip_sign=args.flip_sign,
            keep_input_position=args.keep_input_position,
            n_bins=args.n_bins,
            limit_rows=args.limit_rows_per_fold,
        )
    )

    write_combined_global_table(outputs, out_dir)

    dataset_plot_counts: dict[str, tuple[int, int]] = {}
    if args.plots != "none":
        for item in outputs:
            is_average = item.label == "average"
            should_plot = args.plots == "all" or (args.plots == "average-only" and is_average) or (
                args.plots == "folds-only" and not is_average
            )
            if not should_plot:
                continue
            if is_average:
                plot_root = out_dir / "plots" / "average"
            else:
                plot_root = out_dir / "plots" / "per_fold" / item.label
            dataset_plot_counts[item.label] = plot_dataset(
                item,
                plot_root=plot_root,
                flip_sign=args.flip_sign,
                keep_input_position=args.keep_input_position,
                dpi=args.dpi,
            )

    write_readme(
        out_dir,
        fold_dirs=fold_dirs,
        average_stats=average_stats,
        n_bins=args.n_bins,
        flip_sign=args.flip_sign,
        keep_input_position=args.keep_input_position,
        plots_mode=args.plots,
        dataset_plot_counts=dataset_plot_counts,
        limit_rows=args.limit_rows_per_fold,
        average_method=args.average_method,
    )

    print(f"Done. Wrote README and outputs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
