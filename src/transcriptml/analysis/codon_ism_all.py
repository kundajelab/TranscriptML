#!/usr/bin/env python3
"""Summarize and plot all-codon ISM outputs across Saluki folds.

This script mirrors explore_synonymous_mutations.py for the all-codon ISM
outputs written either as chunked NPZ files under
all_codon_ism/fold*/shard*/mutations_npz/ or as one mutations.parquet file per
shard.

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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-all-codon-ism"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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

LOAD_COLUMNS = IDENTITY_COLUMNS + ["delta"]
AGG_COLUMNS = ["sum_delta", "sum_effect", "sumsq_effect", "n_mutation_rows"]
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
CODON_ORDER = tuple(codon for codons in CODONS_BY_AA.values() for codon in codons)
CODON_SORT = {codon: i for i, codon in enumerate(CODON_ORDER)}

REFERENCE_KEYS = ["reference_amino_acid", "reference_codon"]
PAIR_KEYS = [
    "reference_amino_acid",
    "reference_codon",
    "alternate_amino_acid",
    "alternate_codon",
    "synonymous",
]
REFERENCE_POSITION_KEYS = REFERENCE_KEYS + ["position_bin"]
PAIR_POSITION_KEYS = PAIR_KEYS + ["position_bin"]
AA_TRANSITION_KEYS = ["reference_amino_acid", "alternate_amino_acid", "synonymous"]


@dataclass(frozen=True)
class PartInfo:
    path: str
    n_rows: int
    format: str


@dataclass(frozen=True)
class DatasetOutputs:
    label: str
    reference_table: pl.DataFrame
    pair_table: pl.DataFrame
    reference_position_table: pl.DataFrame
    pair_position_table: pl.DataFrame
    aa_transition_table: pl.DataFrame


def natural_numbered_key(prefix: str, path: Path) -> tuple[int, str]:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def natural_fold_key(path: Path) -> tuple[int, str]:
    return natural_numbered_key("fold", path)


def natural_shard_key(path: Path) -> tuple[int, str]:
    return natural_numbered_key("shard", path)


def default_base_dir() -> Path:
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "Scripts":
        return script_path.parent.parent
    return script_path.parent.parent


def candidate_input_dirs() -> list[Path]:
    candidates = []
    for root in [Path.cwd(), default_base_dir()]:
        candidates.append(root / "all_codon_ism")
        candidates.extend(sorted((root / "Codon_Optimality").glob("*/all_codon_ism")))
        candidates.extend(sorted(root.glob("*/all_codon_ism")))
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize all-codon ISM chunked NPZ or shard parquet files, average "
            "effects across folds by row-aligned mutation keys, and create global "
            "and positional plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fold-dirs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Fold directories containing shard*/mutations_npz/manifest.json or "
            "shard*/mutations.parquet. Overrides --input-dir."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing fold[0-9] all-codon ISM outputs, for example "
            "<run>/all_codon_ism. If omitted, the script "
            "auto-detects only when exactly one candidate exists."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for tables, plots, and README. If omitted, writes to "
            "<dataset-dir>/all_codon_ism_summary when --input-dir or fold dirs "
            "identify a dataset."
        ),
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of equal-width relative CDS position bins.",
    )
    parser.add_argument(
        "--datasets",
        choices=("all", "average-only", "folds-only"),
        default="all",
        help="Which datasets to summarize.",
    )
    parser.add_argument(
        "--plots",
        choices=("all", "average-only", "folds-only", "none"),
        default="all",
        help="Which positional plot sets to render.",
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
        "--verify-row-identity",
        choices=("full", "first-part-per-shard", "none"),
        default="first-part-per-shard",
        help=(
            "How strictly to verify that fold mutation rows are aligned before "
            "fold averaging. Full is safest but adds extra string-array comparisons."
        ),
    )
    parser.add_argument(
        "--summary-combine-chunks",
        type=int,
        default=4,
        help=(
            "Number of chunk aggregate frames to buffer before recompressing summaries. "
            "Lower values use less peak memory."
        ),
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=100_000,
        help="Rows per streamed batch when shard mutation tables are parquet files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N NPZ parts or parquet batches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG plot resolution.",
    )
    parser.add_argument(
        "--max-shards-per-fold",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-parts-per-shard",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--limit-rows-per-part",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.n_bins <= 0:
        parser.error("--n-bins must be positive")
    if args.summary_combine_chunks <= 0:
        parser.error("--summary-combine-chunks must be positive")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.max_shards_per_fold is not None and args.max_shards_per_fold <= 0:
        parser.error("--max-shards-per-fold must be positive when provided")
    if args.max_parts_per_shard is not None and args.max_parts_per_shard <= 0:
        parser.error("--max-parts-per-shard must be positive when provided")
    if args.limit_rows_per_part is not None and args.limit_rows_per_part <= 0:
        parser.error("--limit-rows-per-part must be positive when provided")
    return args


def resolve_input_dir(input_dir: Path | None) -> Path:
    if input_dir is not None:
        resolved = input_dir.resolve()
        if not resolved.exists():
            raise SystemExit(f"--input-dir does not exist: {resolved}")
        return resolved

    candidates = candidate_input_dirs()
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            "No all_codon_ism input directory was found. Pass --input-dir or --fold-dirs."
        )
    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise SystemExit(
        "Multiple all_codon_ism input directories were found; pass --input-dir explicitly.\n"
        f"Candidates:\n{candidate_text}"
    )


def resolve_fold_dirs(fold_dirs: list[Path] | None, input_dir: Path | None) -> tuple[list[Path], Path]:
    if fold_dirs:
        resolved = [path.resolve() for path in fold_dirs]
        parents = {path.parent.resolve() for path in resolved}
        root = next(iter(parents)) if len(parents) == 1 else Path.cwd().resolve()
    else:
        root = resolve_input_dir(input_dir)
        resolved = sorted(root.glob("fold[0-9]"), key=natural_fold_key)

    if not resolved:
        raise SystemExit("No fold directories were found or provided.")

    missing = []
    for fold_dir in resolved:
        shard_dirs = sorted(fold_dir.glob("shard[0-9]*"), key=natural_shard_key)
        if not shard_dirs:
            missing.append(str(fold_dir / "shard*/{mutations.parquet,mutations_npz/manifest.json}"))
            continue
        for shard_dir in shard_dirs:
            manifest = shard_dir / "mutations_npz" / "manifest.json"
            parquet = shard_dir / "mutations.parquet"
            if not manifest.exists() and not parquet.exists():
                missing.append(str(shard_dir / "{mutations.parquet,mutations_npz/manifest.json}"))
    if missing:
        raise SystemExit("Missing all-codon mutation table files:\n" + "\n".join(missing[:20]))

    return sorted(resolved, key=natural_fold_key), root


# Only relevant for NPZ input
def read_manifest(path: Path) -> list[PartInfo]:
    data = json.loads(path.read_text(encoding="utf-8"))
    columns = data.get("columns", [])
    missing = [column for column in LOAD_COLUMNS if column not in columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    parts = [
        PartInfo(path=str(item["path"]), n_rows=int(item["n_rows"]), format="npz")
        for item in data.get("parts", [])
    ]
    if not parts:
        raise SystemExit(f"{path} has no NPZ parts")
    return parts


# Only relevant for parquet input
def read_parquet_part(path: Path) -> list[PartInfo]:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in LOAD_COLUMNS if column not in schema]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    n_rows = int(pq.ParquetFile(path).metadata.num_rows)
    return [PartInfo(path=path.name, n_rows=n_rows, format="parquet")]


# Just figuring out what kind of table(s) we are dealing with
def load_layout(
    fold_dirs: list[Path],
    *,
    max_shards_per_fold: int | None,
    max_parts_per_shard: int | None,
) -> dict[str, dict[str, list[PartInfo]]]:
    layout: dict[str, dict[str, list[PartInfo]]] = {}
    for fold_dir in fold_dirs:
        shard_dirs = sorted(fold_dir.glob("shard[0-9]*"), key=natural_shard_key)
        if max_shards_per_fold is not None:
            shard_dirs = shard_dirs[:max_shards_per_fold]
        fold_layout: dict[str, list[PartInfo]] = {}
        for shard_dir in shard_dirs:
            manifest = shard_dir / "mutations_npz" / "manifest.json"
            parquet = shard_dir / "mutations.parquet"
            if manifest.exists():
                parts = read_manifest(manifest)
                if max_parts_per_shard is not None:
                    parts = parts[:max_parts_per_shard]
            elif parquet.exists():
                parts = read_parquet_part(parquet)
            else:
                raise SystemExit(f"{shard_dir} has neither mutations.parquet nor mutations_npz/manifest.json")
            if not parts:
                raise SystemExit(f"{shard_dir} has no selected mutation parts")
            fold_layout[shard_dir.name] = parts
        layout[fold_dir.name] = fold_layout

    reference_fold = fold_dirs[0].name
    reference_shards = list(layout[reference_fold])
    for fold_dir in fold_dirs[1:]:
        fold_shards = list(layout[fold_dir.name])
        if fold_shards != reference_shards:
            raise SystemExit(
                f"{fold_dir.name} shard layout differs from {reference_fold}: "
                f"{fold_shards} vs {reference_shards}"
            )
        for shard_name in reference_shards:
            ref_parts = layout[reference_fold][shard_name]
            parts = layout[fold_dir.name][shard_name]
            ref_formats = [part.format for part in ref_parts]
            formats = [part.format for part in parts]
            if formats != ref_formats:
                raise SystemExit(f"{fold_dir.name}/{shard_name} table format differs from {reference_fold}")
            ref_paths = [part.path for part in ref_parts]
            paths = [part.path for part in parts]
            if paths != ref_paths:
                raise SystemExit(f"{fold_dir.name}/{shard_name} part names differ from {reference_fold}")
            ref_rows = [part.n_rows for part in ref_parts]
            rows = [part.n_rows for part in parts]
            if rows != ref_rows:
                raise SystemExit(f"{fold_dir.name}/{shard_name} part row counts differ from {reference_fold}")
    return layout


def selected_row_count(
    layout: dict[str, dict[str, list[PartInfo]]],
    fold_name: str,
    *,
    limit_rows_per_part: int | None,
    parquet_batch_size: int,
    max_parts_per_shard: int | None,
) -> tuple[int, int, int]:
    n_shards = len(layout[fold_name])
    n_parts = 0
    n_rows = 0
    for parts in layout[fold_name].values():
        if parts and parts[0].format == "parquet":
            part = parts[0]
            n_batches = math.ceil(part.n_rows / parquet_batch_size)
            if max_parts_per_shard is not None:
                n_batches = min(n_batches, max_parts_per_shard)
            n_parts += n_batches
            for batch_index in range(n_batches):
                remaining = part.n_rows - batch_index * parquet_batch_size
                batch_rows = min(parquet_batch_size, remaining)
                n_rows += min(batch_rows, limit_rows_per_part) if limit_rows_per_part is not None else batch_rows
        else:
            n_parts += len(parts)
            for part in parts:
                n_rows += min(part.n_rows, limit_rows_per_part) if limit_rows_per_part is not None else part.n_rows
    return n_shards, n_parts, n_rows


def layout_input_format(layout: dict[str, dict[str, list[PartInfo]]], fold_name: str) -> str:
    formats = {
        part.format
        for parts in layout[fold_name].values()
        for part in parts
    }
    if len(formats) != 1:
        raise SystemExit(f"Mixed mutation table formats are not supported in one run: {sorted(formats)}")
    return formats.pop()


def load_npz_part(path: Path, *, limit_rows: int | None) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [column for column in LOAD_COLUMNS if column not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        arrays = {}
        for column in LOAD_COLUMNS:
            values = data[column]
            arrays[column] = values[:limit_rows] if limit_rows is not None else values
    return arrays


def arrays_to_frame(arrays: dict[str, np.ndarray]) -> pl.DataFrame:
    return pl.DataFrame({column: arrays[column] for column in LOAD_COLUMNS})


def arrow_batch_to_frame(batch: pa.RecordBatch, *, limit_rows: int | None) -> pl.DataFrame:
    if limit_rows is not None and batch.num_rows > limit_rows:
        batch = batch.slice(0, limit_rows)
    return pl.from_arrow(batch).select(LOAD_COLUMNS)


def verify_identity(
    reference: dict[str, np.ndarray],
    other: dict[str, np.ndarray],
    *,
    fold_name: str,
    shard_name: str,
    part_name: str,
) -> None:
    ref_rows = int(reference["delta"].shape[0])
    other_rows = int(other["delta"].shape[0])
    if ref_rows != other_rows:
        raise ValueError(
            f"row-count mismatch for {fold_name}/{shard_name}/{part_name}: "
            f"{other_rows} vs {ref_rows}"
        )
    for column in IDENTITY_COLUMNS:
        if not np.array_equal(reference[column], other[column]):
            raise ValueError(
                f"identity mismatch for {fold_name}/{shard_name}/{part_name} "
                f"in column {column}"
            )


def verify_identity_frame(
    reference: pl.DataFrame,
    other: pl.DataFrame,
    *,
    fold_name: str,
    shard_name: str,
    part_name: str,
) -> None:
    if reference.height != other.height:
        raise ValueError(
            f"row-count mismatch for {fold_name}/{shard_name}/{part_name}: "
            f"{other.height} vs {reference.height}"
        )
    for column in IDENTITY_COLUMNS:
        if not reference.get_column(column).equals(other.get_column(column)):
            raise ValueError(
                f"identity mismatch for {fold_name}/{shard_name}/{part_name} "
                f"in column {column}"
            )


def aggregate_frame(data: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return data.group_by(keys, maintain_order=False).agg(
        [
            pl.col("delta").sum().alias("sum_delta"),
            pl.col("effect").sum().alias("sum_effect"),
            (pl.col("effect") * pl.col("effect")).sum().alias("sumsq_effect"),
            pl.len().alias("n_mutation_rows"),
        ]
    )


def compress_frames(frames: list[pl.DataFrame], keys: list[str]) -> list[pl.DataFrame]:
    if not frames:
        return []
    combined = pl.concat(frames, how="vertical_relaxed")
    compressed = combined.group_by(keys, maintain_order=False).agg(
        [pl.col(column).sum().alias(column) for column in AGG_COLUMNS]
    )
    return [compressed]


class StreamingSummary:
    def __init__(
        self,
        *,
        label: str,
        flip_sign: bool,
        keep_input_position: bool,
        n_bins: int,
        combine_chunks: int,
    ):
        self.label = label
        self.flip_sign = flip_sign
        self.keep_input_position = keep_input_position
        self.n_bins = int(n_bins)
        self.combine_chunks = int(combine_chunks)
        self.reference_buffers: list[pl.DataFrame] = []
        self.pair_buffers: list[pl.DataFrame] = []
        self.reference_position_buffers: list[pl.DataFrame] = []
        self.pair_position_buffers: list[pl.DataFrame] = []
        self.aa_transition_buffers: list[pl.DataFrame] = []
        self._chunks_since_combine = 0
        self.n_rows = 0

    def add_chunk(self, arrays: dict[str, np.ndarray], *, delta_override: np.ndarray | None = None) -> None:
        self.add_frame(arrays_to_frame(arrays), delta_override=delta_override)

    def add_frame(self, frame: pl.DataFrame, *, delta_override: np.ndarray | None = None) -> None:
        if frame.height == 0:
            return
        if delta_override is not None:
            if delta_override.shape[0] != frame.height:
                raise ValueError(
                    f"delta_override length {delta_override.shape[0]} does not match frame height {frame.height}"
                )
            delta_column = pl.Series("delta", np.asarray(delta_override, dtype=np.float64))
        else:
            delta_column = pl.col("delta").cast(pl.Float64).alias("delta")

        input_position = pl.col("cds_relative_position").cast(pl.Float64)
        position = input_position if self.keep_input_position else (1.0 - input_position)
        effect = -pl.col("delta") if self.flip_sign else pl.col("delta")
        data = frame.with_columns(
            [
                delta_column,
                position.clip(0.0, 1.0).alias(POSITION_COLUMN),
            ]
        ).with_columns(
            [
                effect.alias("effect"),
                (pl.col(POSITION_COLUMN) * self.n_bins)
                .floor()
                .clip(0, self.n_bins - 1)
                .cast(pl.Int32)
                .alias("position_bin"),
            ]
        ).select(
            [
                "reference_amino_acid",
                "reference_codon",
                "alternate_amino_acid",
                "alternate_codon",
                "synonymous",
                "position_bin",
                "delta",
                "effect",
            ]
        )

        self.reference_buffers.append(aggregate_frame(data, REFERENCE_KEYS))
        self.pair_buffers.append(aggregate_frame(data, PAIR_KEYS))
        self.reference_position_buffers.append(aggregate_frame(data, REFERENCE_POSITION_KEYS))
        self.pair_position_buffers.append(aggregate_frame(data, PAIR_POSITION_KEYS))
        self.aa_transition_buffers.append(aggregate_frame(data, AA_TRANSITION_KEYS))
        self.n_rows += int(frame.height)
        self._chunks_since_combine += 1
        if self._chunks_since_combine >= self.combine_chunks:
            self.compress()

    def compress(self) -> None:
        self.reference_buffers = compress_frames(self.reference_buffers, REFERENCE_KEYS)
        self.pair_buffers = compress_frames(self.pair_buffers, PAIR_KEYS)
        self.reference_position_buffers = compress_frames(self.reference_position_buffers, REFERENCE_POSITION_KEYS)
        self.pair_position_buffers = compress_frames(self.pair_position_buffers, PAIR_POSITION_KEYS)
        self.aa_transition_buffers = compress_frames(self.aa_transition_buffers, AA_TRANSITION_KEYS)
        self._chunks_since_combine = 0

    def finalize(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.compress()
        return (
            finalize_aggregate(self.reference_buffers[0] if self.reference_buffers else pl.DataFrame()),
            finalize_aggregate(self.pair_buffers[0] if self.pair_buffers else pl.DataFrame()),
            finalize_aggregate(
                self.reference_position_buffers[0] if self.reference_position_buffers else pl.DataFrame()
            ),
            finalize_aggregate(self.pair_position_buffers[0] if self.pair_position_buffers else pl.DataFrame()),
            finalize_aggregate(self.aa_transition_buffers[0] if self.aa_transition_buffers else pl.DataFrame()),
        )


def finalize_aggregate(data: pl.DataFrame) -> pl.DataFrame:
    if data.is_empty():
        return data
    n = pl.col("n_mutation_rows").cast(pl.Float64)
    mean_effect = pl.col("sum_effect") / n
    raw_variance = (pl.col("sumsq_effect") - (pl.col("sum_effect") * pl.col("sum_effect") / n)) / (n - 1.0)
    variance = pl.when(raw_variance < 0.0).then(0.0).otherwise(raw_variance)
    return data.with_columns(
        [
            (pl.col("sum_delta") / n).alias("mean_delta"),
            mean_effect.alias("mean_effect"),
            pl.when(pl.col("n_mutation_rows") > 1).then(variance.sqrt()).otherwise(None).alias("effect_std"),
        ]
    ).with_columns(
        (pl.col("effect_std") / n.sqrt()).alias("effect_sem")
    )


def add_bin_columns(data: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    if data.is_empty() or "position_bin" not in data.columns:
        return data
    return data.with_columns(
        [
            (pl.col("position_bin").cast(pl.Float64) / n_bins).alias("bin_start"),
            ((pl.col("position_bin").cast(pl.Float64) + 1.0) / n_bins).alias("bin_end"),
            ((pl.col("position_bin").cast(pl.Float64) + 0.5) / n_bins).alias("bin_center"),
        ]
    )


def with_order_columns(data: pl.DataFrame) -> pl.DataFrame:
    columns = []
    if "reference_amino_acid" in data.columns:
        aa_values = data.get_column("reference_amino_acid").to_list()
        columns.append(
            pl.Series("reference_amino_acid_order", [AA_SORT.get(value, 10**9) for value in aa_values], dtype=pl.Int64)
        )
    if "alternate_amino_acid" in data.columns:
        aa_values = data.get_column("alternate_amino_acid").to_list()
        columns.append(
            pl.Series("alternate_amino_acid_order", [AA_SORT.get(value, 10**9) for value in aa_values], dtype=pl.Int64)
        )
    if "reference_codon" in data.columns:
        codon_values = data.get_column("reference_codon").to_list()
        columns.append(
            pl.Series("reference_codon_order", [CODON_SORT.get(value, 10**9) for value in codon_values], dtype=pl.Int64)
        )
    if "alternate_codon" in data.columns:
        codon_values = data.get_column("alternate_codon").to_list()
        columns.append(
            pl.Series("alternate_codon_order", [CODON_SORT.get(value, 10**9) for value in codon_values], dtype=pl.Int64)
        )
    return data.with_columns(columns) if columns else data


def drop_order_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.drop([column for column in data.columns if column.endswith("_order")])


def add_amino_acid_names(data: pl.DataFrame) -> pl.DataFrame:
    columns = []
    if "reference_amino_acid" in data.columns:
        values = data.get_column("reference_amino_acid").to_list()
        columns.append(pl.Series("amino_acid_name", [AA_NAMES.get(value, value) for value in values], dtype=pl.String))
    if "alternate_amino_acid" in data.columns:
        values = data.get_column("alternate_amino_acid").to_list()
        columns.append(
            pl.Series("alternate_amino_acid_name", [AA_NAMES.get(value, value) for value in values], dtype=pl.String)
        )
    return data.with_columns(columns) if columns else data


def add_reference_instance_counts(reference: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    if reference.is_empty() or pair.is_empty():
        return reference
    alt_counts = pair.group_by(REFERENCE_KEYS).agg(
        pl.col("alternate_codon").n_unique().alias("n_alternate_codons")
    )
    return reference.join(alt_counts, on=REFERENCE_KEYS, how="left").with_columns(
        (
            pl.col("n_mutation_rows").cast(pl.Float64)
            / pl.col("n_alternate_codons").cast(pl.Float64)
        ).alias("n_unique_sequence_codon_instances")
    )


def add_pair_instance_counts(pair: pl.DataFrame) -> pl.DataFrame:
    if pair.is_empty():
        return pair
    return pair.with_columns(
        pl.col("n_mutation_rows").cast(pl.Float64).alias("n_unique_sequence_codon_instances")
    )


def add_aa_transition_instance_counts(aa_transition: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    if aa_transition.is_empty() or pair.is_empty():
        return aa_transition
    alt_counts = pair.group_by(AA_TRANSITION_KEYS).agg(
        pl.col("alternate_codon").n_unique().alias("n_alternate_codons")
    )
    return aa_transition.join(alt_counts, on=AA_TRANSITION_KEYS, how="left").with_columns(
        (
            pl.col("n_mutation_rows").cast(pl.Float64)
            / pl.col("n_alternate_codons").cast(pl.Float64)
        ).alias("n_unique_sequence_codon_instances")
    )


def rank_and_order_reference(data: pl.DataFrame) -> pl.DataFrame:
    data = with_order_columns(add_amino_acid_names(data))
    data = data.with_columns(
        pl.col("mean_effect")
        .rank(method="ordinal", descending=True)
        .over("reference_amino_acid")
        .cast(pl.Int64)
        .alias("within_amino_acid_rank")
    )
    data = data.sort(["reference_amino_acid_order", "within_amino_acid_rank", "reference_codon_order"])
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
        "n_alternate_codons",
        "within_amino_acid_rank",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def order_pair_table(data: pl.DataFrame) -> pl.DataFrame:
    data = with_order_columns(add_amino_acid_names(data))
    data = data.with_columns(
        pl.col("mean_effect")
        .rank(method="ordinal", descending=True)
        .over(["reference_amino_acid", "reference_codon"])
        .cast(pl.Int64)
        .alias("within_reference_codon_rank")
    )
    data = data.sort(
        [
            "reference_amino_acid_order",
            "reference_codon_order",
            "within_reference_codon_rank",
            "alternate_amino_acid_order",
            "alternate_codon_order",
        ]
    )
    ordered_columns = [
        "reference_amino_acid",
        "amino_acid_name",
        "reference_codon",
        "alternate_amino_acid",
        "alternate_amino_acid_name",
        "alternate_codon",
        "synonymous",
        "mean_delta",
        "mean_effect",
        "effect_std",
        "effect_sem",
        "n_mutation_rows",
        "n_unique_sequence_codon_instances",
        "within_reference_codon_rank",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def order_reference_position(data: pl.DataFrame, pair_position: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    if not data.is_empty() and not pair_position.is_empty():
        alt_counts = pair_position.group_by(REFERENCE_POSITION_KEYS).agg(
            pl.col("alternate_codon").n_unique().alias("n_alternate_codons")
        )
        data = data.join(alt_counts, on=REFERENCE_POSITION_KEYS, how="left").with_columns(
            (
                pl.col("n_mutation_rows").cast(pl.Float64)
                / pl.col("n_alternate_codons").cast(pl.Float64)
            ).alias("n_unique_sequence_codon_instances")
        )
    data = add_bin_columns(with_order_columns(add_amino_acid_names(data)), n_bins)
    data = data.sort(["reference_amino_acid_order", "reference_codon_order", "position_bin"])
    ordered_columns = [
        "reference_amino_acid",
        "amino_acid_name",
        "reference_codon",
        "position_bin",
        "bin_start",
        "bin_end",
        "bin_center",
        "mean_delta",
        "mean_effect",
        "effect_std",
        "effect_sem",
        "n_mutation_rows",
        "n_unique_sequence_codon_instances",
        "n_alternate_codons",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def order_pair_position(data: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    data = add_pair_instance_counts(data)
    data = add_bin_columns(with_order_columns(add_amino_acid_names(data)), n_bins)
    data = data.sort(
        [
            "reference_amino_acid_order",
            "reference_codon_order",
            "alternate_amino_acid_order",
            "alternate_codon_order",
            "position_bin",
        ]
    )
    ordered_columns = [
        "reference_amino_acid",
        "amino_acid_name",
        "reference_codon",
        "alternate_amino_acid",
        "alternate_amino_acid_name",
        "alternate_codon",
        "synonymous",
        "position_bin",
        "bin_start",
        "bin_end",
        "bin_center",
        "mean_delta",
        "mean_effect",
        "effect_std",
        "effect_sem",
        "n_mutation_rows",
        "n_unique_sequence_codon_instances",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def order_aa_transition(data: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    data = add_aa_transition_instance_counts(data, pair)
    data = with_order_columns(add_amino_acid_names(data))
    data = data.sort(["reference_amino_acid_order", "alternate_amino_acid_order"])
    ordered_columns = [
        "reference_amino_acid",
        "amino_acid_name",
        "alternate_amino_acid",
        "alternate_amino_acid_name",
        "synonymous",
        "mean_delta",
        "mean_effect",
        "effect_std",
        "effect_sem",
        "n_mutation_rows",
        "n_unique_sequence_codon_instances",
        "n_alternate_codons",
    ]
    return drop_order_columns(data.select([column for column in ordered_columns if column in data.columns]))


def format_outputs(summary: StreamingSummary, *, table_dir: Path, n_bins: int) -> DatasetOutputs:
    print(f"[{summary.label}] Finalizing summary tables ({summary.n_rows} rows)", flush=True)
    reference, pair, reference_position, pair_position, aa_transition = summary.finalize()
    reference = add_reference_instance_counts(reference, pair)
    pair = add_pair_instance_counts(pair)

    reference = rank_and_order_reference(reference)
    pair = order_pair_table(pair)
    reference_position = order_reference_position(reference_position, pair_position, n_bins)
    pair_position = order_pair_position(pair_position, n_bins)
    aa_transition = order_aa_transition(aa_transition, pair)

    write_table(reference, table_dir / "global_reference_codon_effects")
    write_table(pair, table_dir / "global_codon_pair_effects")
    write_table(reference_position, table_dir / "position_bins_by_reference_codon")
    write_table(pair_position, table_dir / "position_bins_by_codon_pair")
    write_table(aa_transition, table_dir / "amino_acid_transition_effects")

    return DatasetOutputs(
        label=summary.label,
        reference_table=reference,
        pair_table=pair,
        reference_position_table=reference_position,
        pair_position_table=pair_position,
        aa_transition_table=aa_transition,
    )


def write_table(data: pl.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(stem.with_suffix(".csv"))
    data.write_csv(stem.with_suffix(".tsv"), separator="\t")


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
            f"codon(s): {', '.join(missing)}. This can happen in bounded debug runs "
            "or when those reference codons are absent from the dataset."
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
    reference_position: pl.DataFrame,
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
        subset = reference_position.filter(pl.col("reference_amino_acid") == aa)
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


def diverging_norm(values: np.ndarray) -> TwoSlopeNorm | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin >= 0.0 or vmax <= 0.0:
        return None
    limit = max(abs(vmin), abs(vmax))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def draw_heatmap(
    matrix,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    width: float,
    height: float,
) -> plt.Figure:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0.0,
        norm=diverging_norm(values),
        cbar_kws={"label": colorbar_label},
        linewidths=0.0,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    fig.tight_layout()
    return fig


def plot_by_reference_codon_heatmaps(
    pair_position: pl.DataFrame,
    *,
    out_dir: Path,
    label: str,
    flip_sign: bool,
    keep_input_position: bool,
    n_bins: int,
    dpi: int,
) -> int:
    setup_plot_style()
    n_plots = 0
    y_label = effect_axis_label(flip_sign)
    x_label = position_axis_label(keep_input_position)

    for aa in AA_ORDER:
        for codon in CODONS_BY_AA[aa]:
            subset = pair_position.filter(
                (pl.col("reference_amino_acid") == aa) & (pl.col("reference_codon") == codon)
            )
            if subset.is_empty():
                continue
            observed_alts = set(subset.get_column("alternate_codon").unique().to_list())
            alt_order = [alt for alt in CODON_ORDER if alt in observed_alts and alt != codon]
            if not alt_order:
                continue
            pdf = subset.to_pandas()
            matrix = pdf.pivot_table(
                index="alternate_codon",
                columns="position_bin",
                values="mean_effect",
                aggfunc="mean",
            )
            matrix = matrix.reindex(index=alt_order, columns=list(range(n_bins)))
            matrix.columns = [f"{(i + 0.5) / n_bins:.2f}" for i in range(n_bins)]
            title = f"{AA_NAMES.get(aa, aa)} ({aa}) {codon} alternate-codon effects - {label}"
            fig = draw_heatmap(
                matrix,
                title=title,
                xlabel=x_label,
                ylabel="Alternate codon",
                colorbar_label=y_label,
                width=8.6,
                height=10.5,
            )
            stem = out_dir / f"{safe_filename(aa)}_{safe_filename(codon)}"
            save_figure(fig, stem, dpi)
            n_plots += 1
    return n_plots


def plot_global_codon_heatmap(
    pair_table: pl.DataFrame,
    *,
    out_dir: Path,
    label: str,
    flip_sign: bool,
    dpi: int,
) -> int:
    if pair_table.is_empty():
        return 0
    setup_plot_style()
    pdf = pair_table.to_pandas()
    matrix = pdf.pivot_table(
        index="reference_codon",
        columns="alternate_codon",
        values="mean_effect",
        aggfunc="mean",
    )
    matrix = matrix.reindex(index=list(CODON_ORDER), columns=list(CODON_ORDER))
    title = f"All-codon ISM codon-to-codon effects - {label}"
    fig = draw_heatmap(
        matrix,
        title=title,
        xlabel="Alternate codon",
        ylabel="Reference codon",
        colorbar_label=effect_axis_label(flip_sign),
        width=13.5,
        height=12.0,
    )
    save_figure(fig, out_dir / "codon_to_codon_effect_heatmap", dpi)
    return 1


def plot_amino_acid_transition_heatmap(
    aa_transition: pl.DataFrame,
    *,
    out_dir: Path,
    label: str,
    flip_sign: bool,
    dpi: int,
) -> int:
    if aa_transition.is_empty():
        return 0
    setup_plot_style()
    pdf = aa_transition.to_pandas()
    matrix = pdf.pivot_table(
        index="reference_amino_acid",
        columns="alternate_amino_acid",
        values="mean_effect",
        aggfunc="mean",
    )
    matrix = matrix.reindex(index=list(AA_ORDER), columns=list(AA_ORDER))
    title = f"All-codon ISM amino-acid transition effects - {label}"
    fig = draw_heatmap(
        matrix,
        title=title,
        xlabel="Alternate amino acid",
        ylabel="Reference amino acid",
        colorbar_label=effect_axis_label(flip_sign),
        width=8.5,
        height=7.8,
    )
    save_figure(fig, out_dir / "amino_acid_transition_effect_heatmap", dpi)
    return 1


def plot_dataset(
    outputs: DatasetOutputs,
    *,
    plot_root: Path,
    flip_sign: bool,
    keep_input_position: bool,
    n_bins: int,
    dpi: int,
) -> dict[str, int]:
    print(f"[{outputs.label}] Rendering plots", flush=True)
    counts = {
        "amino_acid_lineplots": plot_by_amino_acid(
            outputs.reference_position_table,
            out_dir=plot_root / "by_amino_acid",
            label=outputs.label,
            flip_sign=flip_sign,
            keep_input_position=keep_input_position,
            dpi=dpi,
        ),
        "reference_codon_heatmaps": plot_by_reference_codon_heatmaps(
            outputs.pair_position_table,
            out_dir=plot_root / "by_reference_codon",
            label=outputs.label,
            flip_sign=flip_sign,
            keep_input_position=keep_input_position,
            n_bins=n_bins,
            dpi=dpi,
        ),
        "global_codon_heatmaps": plot_global_codon_heatmap(
            outputs.pair_table,
            out_dir=plot_root / "global",
            label=outputs.label,
            flip_sign=flip_sign,
            dpi=dpi,
        ),
        "amino_acid_transition_heatmaps": plot_amino_acid_transition_heatmap(
            outputs.aa_transition_table,
            out_dir=plot_root / "global",
            label=outputs.label,
            flip_sign=flip_sign,
            dpi=dpi,
        ),
    }
    return counts


def write_combined_tables(outputs: list[DatasetOutputs], out_dir: Path) -> None:
    if not outputs:
        return
    reference_frames = [item.reference_table.with_columns(pl.lit(item.label).alias("dataset")) for item in outputs]
    reference = pl.concat(reference_frames, how="vertical_relaxed")
    write_table(reference.select(["dataset"] + [column for column in reference.columns if column != "dataset"]), out_dir / "tables" / "global_reference_codon_effects_by_dataset")

    pair_frames = [item.pair_table.with_columns(pl.lit(item.label).alias("dataset")) for item in outputs]
    pair = pl.concat(pair_frames, how="vertical_relaxed")
    write_table(pair.select(["dataset"] + [column for column in pair.columns if column != "dataset"]), out_dir / "tables" / "global_codon_pair_effects_by_dataset")

    transition_frames = [item.aa_transition_table.with_columns(pl.lit(item.label).alias("dataset")) for item in outputs]
    transition = pl.concat(transition_frames, how="vertical_relaxed")
    write_table(transition.select(["dataset"] + [column for column in transition.columns if column != "dataset"]), out_dir / "tables" / "amino_acid_transition_effects_by_dataset")


def should_verify_part(mode: str, *, part_index: int) -> bool:
    if mode == "full":
        return True
    if mode == "first-part-per-shard":
        return part_index == 0
    return False


def process_npz_inputs(
    fold_dirs: list[Path],
    layout: dict[str, dict[str, list[PartInfo]]],
    *,
    fold_summaries: list[StreamingSummary],
    average_summary: StreamingSummary | None,
    limit_rows_per_part: int | None,
    verify_row_identity: str,
    progress_every: int,
) -> dict[str, int | str]:
    reference_fold = fold_dirs[0]
    shard_names = list(layout[reference_fold.name])
    total_parts = sum(len(layout[reference_fold.name][shard_name]) for shard_name in shard_names)
    processed_parts = 0
    processed_rows_per_fold = 0
    print(
        f"[stream] Processing {len(fold_dirs)} folds, {len(shard_names)} shards, "
        f"{total_parts} parts/fold",
        flush=True,
    )

    for shard_name in shard_names:
        parts = layout[reference_fold.name][shard_name]
        for part_index, part in enumerate(parts):
            part_name = part.path
            base_path = reference_fold / shard_name / "mutations_npz" / part_name
            base = load_npz_part(base_path, limit_rows=limit_rows_per_part)
            n_rows = int(base["delta"].shape[0])
            processed_rows_per_fold += n_rows

            if fold_summaries:
                fold_summaries[0].add_chunk(base)
            delta_sum = np.asarray(base["delta"], dtype=np.float64) if average_summary is not None else None

            verify_this_part = should_verify_part(verify_row_identity, part_index=part_index)
            for fold_offset, fold_dir in enumerate(fold_dirs[1:], start=1):
                other_path = fold_dir / shard_name / "mutations_npz" / part_name
                other = load_npz_part(other_path, limit_rows=limit_rows_per_part)
                if verify_this_part:
                    verify_identity(base, other, fold_name=fold_dir.name, shard_name=shard_name, part_name=part_name)
                if fold_summaries:
                    fold_summaries[fold_offset].add_chunk(other)
                if delta_sum is not None:
                    delta_sum += np.asarray(other["delta"], dtype=np.float64)

            if average_summary is not None and delta_sum is not None:
                average_summary.add_chunk(base, delta_override=delta_sum / float(len(fold_dirs)))

            processed_parts += 1
            if processed_parts % progress_every == 0 or processed_parts == total_parts:
                print(
                    f"[stream] {processed_parts}/{total_parts} parts complete; "
                    f"{processed_rows_per_fold:,} rows/fold streamed",
                    flush=True,
                )

    return {
        "input_format": "npz",
        "n_folds": len(fold_dirs),
        "n_shards": len(shard_names),
        "n_parts_per_fold": total_parts,
        "n_rows_per_fold": processed_rows_per_fold,
        "verify_row_identity": verify_row_identity,
    }

# fold_dirs: list of paths to directories with fold-specific data
# layout: dictionary describing basic details of each file
# fold_summaries: kinda complicated list of classes related to streaming the parquet to avoid RAM-problems
# average_summary: like fold_summaries but for averaged output I guess
# limit_rows_per_part: How many rows to read into RAM at a time
# verify_row_identity: How to confirm that rows being averaged across folds are same
# progress_every: printing stuff
# parquest_batch_size: Actual number of rows to read into RAM at a time? Per fold at least
# max_parts_per_shard: Not sure, but another streaming parameter. Might set upper bound on how many batches get read per
#    fold in total. Like you can use this to just analyze a subset, probably for QC reasons
def process_parquet_inputs(
    fold_dirs: list[Path],
    layout: dict[str, dict[str, list[PartInfo]]],
    *,
    fold_summaries: list[StreamingSummary],
    average_summary: StreamingSummary | None,
    limit_rows_per_part: int | None,
    verify_row_identity: str,
    progress_every: int,
    parquet_batch_size: int,
    max_parts_per_shard: int | None,
) -> dict[str, int | str]:

    # Just figuring how some details about the files that will be streamed
    reference_fold = fold_dirs[0]
    shard_names = list(layout[reference_fold.name])
    total_parts = 0
    for shard_name in shard_names:
        part = layout[reference_fold.name][shard_name][0]
        n_batches = math.ceil(part.n_rows / parquet_batch_size)
        total_parts += min(n_batches, max_parts_per_shard) if max_parts_per_shard is not None else n_batches

    processed_parts = 0
    processed_rows_per_fold = 0
    print(
        f"[stream] Processing {len(fold_dirs)} folds, {len(shard_names)} parquet shards, "
        f"{total_parts} batches/fold",
        flush=True,
    )

    for shard_name in shard_names:
        parquet_paths = [fold_dir / shard_name / "mutations.parquet" for fold_dir in fold_dirs]
        parquet_files = [pq.ParquetFile(path) for path in parquet_paths]
        iterators = [
            parquet_file.iter_batches(batch_size=parquet_batch_size, columns=LOAD_COLUMNS)
            for parquet_file in parquet_files
        ]
        batch_index = 0
        while True:
            if max_parts_per_shard is not None and batch_index >= max_parts_per_shard:
                break
            try:
                base_batch = next(iterators[0])
            except StopIteration:
                break

            part_name = f"mutations.parquet:batch{batch_index}"
            base = arrow_batch_to_frame(base_batch, limit_rows=limit_rows_per_part)
            n_rows = int(base.height)
            processed_rows_per_fold += n_rows

            if fold_summaries:
                fold_summaries[0].add_frame(base)
            delta_sum = (
                base.get_column("delta").cast(pl.Float64).to_numpy().copy()
                if average_summary is not None
                else None
            )

            verify_this_part = should_verify_part(verify_row_identity, part_index=batch_index)
            for fold_offset, fold_dir in enumerate(fold_dirs[1:], start=1):
                try:
                    other_batch = next(iterators[fold_offset])
                except StopIteration as exc:
                    raise ValueError(
                        f"{fold_dir.name}/{shard_name} ended before {reference_fold.name} at {part_name}"
                    ) from exc
                other = arrow_batch_to_frame(other_batch, limit_rows=limit_rows_per_part)
                if verify_this_part:
                    verify_identity_frame(base, other, fold_name=fold_dir.name, shard_name=shard_name, part_name=part_name)
                if fold_summaries:
                    fold_summaries[fold_offset].add_frame(other)
                if delta_sum is not None:
                    delta_sum += other.get_column("delta").cast(pl.Float64).to_numpy()

            if average_summary is not None and delta_sum is not None:
                average_summary.add_frame(base, delta_override=delta_sum / float(len(fold_dirs)))

            processed_parts += 1
            batch_index += 1
            if processed_parts % progress_every == 0 or processed_parts == total_parts:
                print(
                    f"[stream] {processed_parts}/{total_parts} batches complete; "
                    f"{processed_rows_per_fold:,} rows/fold streamed",
                    flush=True,
                )

    return {
        "input_format": "parquet",
        "n_folds": len(fold_dirs),
        "n_shards": len(shard_names),
        "n_parts_per_fold": total_parts,
        "n_rows_per_fold": processed_rows_per_fold,
        "parquet_batch_size": parquet_batch_size,
        "verify_row_identity": verify_row_identity,
    }


def write_readme(
    out_dir: Path,
    *,
    fold_dirs: list[Path],
    stream_stats: dict[str, int | str],
    n_bins: int,
    flip_sign: bool,
    keep_input_position: bool,
    datasets_mode: str,
    plots_mode: str,
    dataset_plot_counts: dict[str, dict[str, int]],
    debug_limits: dict[str, int | str | None],
) -> None:
    sign_line = sign_summary(flip_sign)
    position_line = (
        "Position coordinate: used input cds_relative_position directly."
        if keep_input_position
        else "Position coordinate: converted to cds_position_5p_to_3p = 1 - cds_relative_position before binning."
    )
    plot_lines = []
    for label, counts in sorted(dataset_plot_counts.items()):
        plot_lines.append(
            "- "
            + label
            + ": "
            + ", ".join(f"{value} {key}" for key, value in sorted(counts.items()))
        )
    if not plot_lines:
        plot_lines = ["- positional plot rendering was skipped"]
    folds = "\n".join(f"- {path}" for path in fold_dirs)
    stats_json = json.dumps(stream_stats, indent=2, sort_keys=True)
    debug_json = json.dumps(debug_limits, indent=2, sort_keys=True)
    text = f"""# All-codon ISM summary

{sign_line}

{position_line}

Number of position bins: {n_bins}
Datasets mode: {datasets_mode}
Plots mode: {plots_mode}

Input folds:
{folds}

Tables:
- {out_dir / "tables"}

Plots:
- {out_dir / "plots"}

Streaming input summary:
```json
{stats_json}
```

Debug/input bounds:
```json
{debug_json}
```

Rendered plots:
{chr(10).join(plot_lines)}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    fold_dirs, input_root = resolve_fold_dirs(args.fold_dirs, args.input_dir)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (input_root.parent / "all_codon_ism_summary").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout(
        fold_dirs,
        max_shards_per_fold=args.max_shards_per_fold,
        max_parts_per_shard=args.max_parts_per_shard,
    )
    input_format = layout_input_format(layout, fold_dirs[0].name)
    n_shards, n_parts, n_rows = selected_row_count(
        layout,
        fold_dirs[0].name,
        limit_rows_per_part=args.limit_rows_per_part,
        parquet_batch_size=args.parquet_batch_size,
        max_parts_per_shard=args.max_parts_per_shard,
    )

    print("All-codon ISM summarization", flush=True)
    print(f"Input root: {input_root}", flush=True)
    print(f"Input format: {input_format}", flush=True)
    print(f"Input folds: {', '.join(path.name for path in fold_dirs)}", flush=True)
    part_word = "batches" if input_format == "parquet" else "parts"
    print(f"Selected shards/fold: {n_shards}; selected {part_word}/fold: {n_parts}", flush=True)
    print(f"Selected rows/fold: {n_rows:,}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(sign_summary(args.flip_sign), flush=True)
    print(position_axis_label(args.keep_input_position), flush=True)

    compute_folds = args.datasets in {"all", "folds-only"}
    compute_average = args.datasets in {"all", "average-only"}
    fold_summaries = [
        StreamingSummary(
            label=fold_dir.name,
            flip_sign=args.flip_sign,
            keep_input_position=args.keep_input_position,
            n_bins=args.n_bins,
            combine_chunks=args.summary_combine_chunks,
        )
        for fold_dir in fold_dirs
    ] if compute_folds else []
    average_summary = (
        StreamingSummary(
            label="average",
            flip_sign=args.flip_sign,
            keep_input_position=args.keep_input_position,
            n_bins=args.n_bins,
            combine_chunks=args.summary_combine_chunks,
        )
        if compute_average
        else None
    )

    if input_format == "npz":
        stream_stats = process_npz_inputs(
            fold_dirs,
            layout,
            fold_summaries=fold_summaries,
            average_summary=average_summary,
            limit_rows_per_part=args.limit_rows_per_part,
            verify_row_identity=args.verify_row_identity,
            progress_every=args.progress_every,
        )
    elif input_format == "parquet":
        stream_stats = process_parquet_inputs(
            fold_dirs,
            layout,
            fold_summaries=fold_summaries,
            average_summary=average_summary,
            limit_rows_per_part=args.limit_rows_per_part,
            verify_row_identity=args.verify_row_identity,
            progress_every=args.progress_every,
            parquet_batch_size=args.parquet_batch_size,
            max_parts_per_shard=args.max_parts_per_shard,
        )
    else:
        raise SystemExit(f"Unsupported input format: {input_format}")

    outputs: list[DatasetOutputs] = []
    for summary in fold_summaries:
        outputs.append(format_outputs(summary, table_dir=out_dir / "tables" / "per_fold" / summary.label, n_bins=args.n_bins))
    if average_summary is not None:
        outputs.append(format_outputs(average_summary, table_dir=out_dir / "tables" / "average", n_bins=args.n_bins))

    write_combined_tables(outputs, out_dir)

    dataset_plot_counts: dict[str, dict[str, int]] = {}
    if args.plots != "none":
        for item in outputs:
            is_average = item.label == "average"
            should_plot = args.plots == "all" or (args.plots == "average-only" and is_average) or (
                args.plots == "folds-only" and not is_average
            )
            if not should_plot:
                continue
            plot_root = out_dir / "plots" / ("average" if is_average else f"per_fold/{item.label}")
            dataset_plot_counts[item.label] = plot_dataset(
                item,
                plot_root=plot_root,
                flip_sign=args.flip_sign,
                keep_input_position=args.keep_input_position,
                n_bins=args.n_bins,
                dpi=args.dpi,
            )

    write_readme(
        out_dir,
        fold_dirs=fold_dirs,
        stream_stats=stream_stats,
        n_bins=args.n_bins,
        flip_sign=args.flip_sign,
        keep_input_position=args.keep_input_position,
        datasets_mode=args.datasets,
        plots_mode=args.plots,
        dataset_plot_counts=dataset_plot_counts,
        debug_limits={
            "input_format": input_format,
            "max_shards_per_fold": args.max_shards_per_fold,
            "max_parts_per_shard": args.max_parts_per_shard,
            "limit_rows_per_part": args.limit_rows_per_part,
            "parquet_batch_size": args.parquet_batch_size,
        },
    )

    print(f"Done. Wrote README and outputs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
