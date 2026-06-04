#!/usr/bin/env python3
"""Plot CDS codon usage by relative position from a Saluki-style X.npy.

The input X array is expected to have shape ``(N, 6, L)`` with base channels
A/C/G/U(T), a CDS-codon-start channel, and a splice channel. The script counts
reference codons at observed CDS codon starts, bins them by CDS-relative
position, and normalizes within each amino-acid family so synonymous codon
usage fractions sum to one per amino acid and position bin.

Transcripts are included only when their full CDS appears in X. This is checked
by comparing the number of observed, decodable CDS codon starts against
``metadata.json``'s full ``cds_length // 3`` value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-codon-usage"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
CODON_TO_AA = {codon: aa for aa, codons in CODONS_BY_AA.items() for codon in codons}
CODON_SORT = {
    codon: i for i, codon in enumerate(codon for codons in CODONS_BY_AA.values() for codon in codons)
}
BASE_ORDER = ("A", "C", "G", "U")


def default_data_dir() -> Path:
    """Find the default Training/RDC_TTDB_All_SalukiExact/data directory."""

    relative = Path("Training") / "RDC_TTDB_All_SalukiExact" / "data"
    if relative.exists():
        return relative

    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / relative
        if candidate.exists():
            return candidate

    sibling = script_path.parents[1].parent / "TranscriptML_Exploration" / relative
    if sibling.exists():
        return sibling

    return relative


def default_out_dir(data_dir: Path) -> Path:
    """Choose an output directory near the matching Codon_Optimality run."""

    run_name = data_dir.parent.name
    for parent in data_dir.resolve().parents:
        candidate = parent / "Codon_Optimality" / run_name
        if candidate.exists():
            return candidate / "codon_usage_summary"
    return data_dir.parent / "codon_usage_summary"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot CDS-position-binned synonymous codon usage "
            "fractions from Training/RDC_TTDB_All_SalukiExact/data/X.npy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing X.npy, metadata.json, and schema.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for tables, plots, and README.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of equal-width CDS-relative position bins.",
    )
    parser.add_argument(
        "--plots",
        choices=("all", "none"),
        default="all",
        help="Whether to render per-amino-acid position plots.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG plot resolution.")
    parser.add_argument(
        "--limit-transcripts",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.n_bins <= 0:
        parser.error("--n-bins must be positive")
    if args.limit_transcripts is not None and args.limit_transcripts <= 0:
        parser.error("--limit-transcripts must be positive when provided")
    return args


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_channels(schema: dict[str, object]) -> tuple[np.ndarray, tuple[str, ...], int]:
    channels = list(schema.get("channels", []))
    if not channels:
        return np.asarray([0, 1, 2, 3], dtype=np.int64), BASE_ORDER, 4

    base_indices = []
    base_letters = []
    for base in ("A", "C", "G", "U"):
        candidates = [base]
        if base == "U":
            candidates.append("T")
        index = next((channels.index(name) for name in candidates if name in channels), None)
        if index is None:
            raise SystemExit(f"Could not find base channel {base}/T in schema channels: {channels}")
        base_indices.append(index)
        base_letters.append(base)

    preferred_cds = ("CDS_codon_start", "cds_codon_start", "codon_start", "CDS", "cds")
    lower_to_index = {str(name).lower(): i for i, name in enumerate(channels)}
    cds_index = None
    for name in preferred_cds:
        if name.lower() in lower_to_index:
            cds_index = lower_to_index[name.lower()]
            break
    if cds_index is None:
        for i, name in enumerate(channels):
            lowered = str(name).lower()
            if "codon_start" in lowered or "cds" in lowered:
                cds_index = i
                break
    if cds_index is None:
        raise SystemExit(f"Could not infer CDS codon-start channel from schema channels: {channels}")

    return np.asarray(base_indices, dtype=np.int64), tuple(base_letters), int(cds_index)


def valid_length(seq: np.ndarray, base_indices: np.ndarray) -> int:
    base_present = np.any(seq[base_indices] > 0, axis=0)
    present = np.flatnonzero(base_present)
    if present.size == 0:
        return 0
    return int(present[-1]) + 1


def decode_codons(
    seq: np.ndarray,
    starts: np.ndarray,
    *,
    valid_len: int,
    base_indices: np.ndarray,
    base_letters: Sequence[str],
) -> list[str] | None:
    base_values = seq[base_indices, :valid_len]
    base_calls = np.argmax(base_values, axis=0)
    base_ok = base_values.sum(axis=0) > 0
    codons = []
    for start_raw in starts:
        start = int(start_raw)
        end = start + 3
        if end > valid_len or not np.all(base_ok[start:end]):
            return None
        codon = "".join(base_letters[int(base_calls[pos])] for pos in range(start, end))
        if codon not in CODON_TO_AA:
            return None
        codons.append(codon)
    return codons


def position_bin(codon_index: int, n_codons: int, n_bins: int) -> int:
    raw = int(math.floor((codon_index / n_codons) * n_bins))
    return min(max(raw, 0), n_bins - 1)


def analyze_usage(
    X: np.ndarray,
    metadata: list[dict[str, object]],
    *,
    base_indices: np.ndarray,
    base_letters: Sequence[str],
    cds_index: int,
    n_bins: int,
    limit_transcripts: int | None,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, int, str], int], dict[str, object]]:
    global_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    bin_counts: defaultdict[tuple[str, int, str], int] = defaultdict(int)
    excluded: Counter[str] = Counter()
    included = 0
    total_codons = 0
    n_sequences = int(X.shape[0])
    n_to_scan = min(n_sequences, limit_transcripts) if limit_transcripts is not None else n_sequences

    for seq_i in range(n_to_scan):
        seq = np.asarray(X[seq_i])
        meta = metadata[seq_i]
        cds_length_raw = meta.get("cds_length")
        if cds_length_raw is None:
            excluded["missing_cds_length"] += 1
            continue
        cds_length = int(cds_length_raw)
        if cds_length <= 0:
            excluded["noncoding_or_zero_cds"] += 1
            continue
        if cds_length % 3 != 0:
            excluded["cds_length_not_divisible_by_three"] += 1
            continue

        expected_codons = cds_length // 3
        seq_valid_len = valid_length(seq, base_indices)
        starts = np.flatnonzero(seq[cds_index, :seq_valid_len] > 0)
        if int(starts.size) != expected_codons:
            excluded["incomplete_observed_cds"] += 1
            continue

        codons = decode_codons(
            seq,
            starts,
            valid_len=seq_valid_len,
            base_indices=base_indices,
            base_letters=base_letters,
        )
        if codons is None:
            excluded["undecodable_cds_codon"] += 1
            continue

        included += 1
        total_codons += len(codons)
        for codon_index, codon in enumerate(codons):
            aa = CODON_TO_AA[codon]
            bin_id = position_bin(codon_index, expected_codons, n_bins)
            global_counts[(aa, codon)] += 1
            bin_counts[(aa, bin_id, codon)] += 1

        if included and included % 1000 == 0:
            print(f"[usage] Included {included} full-CDS transcripts", flush=True)

    summary = {
        "n_sequences": n_sequences,
        "n_scanned_sequences": n_to_scan,
        "n_included_full_cds_transcripts": included,
        "n_excluded_transcripts": int(sum(excluded.values())),
        "excluded_transcripts_by_reason": dict(sorted(excluded.items())),
        "n_counted_codons": total_codons,
        "n_bins": n_bins,
        "full_cds_filter": (
            "included only transcripts where observed decodable CDS codon-start "
            "count equals metadata cds_length // 3"
        ),
    }
    return dict(global_counts), dict(bin_counts), summary


def bin_columns(bin_id: int, n_bins: int) -> dict[str, float | int]:
    return {
        "position_bin": bin_id,
        "bin_start": bin_id / n_bins,
        "bin_end": (bin_id + 1) / n_bins,
        "bin_center": (bin_id + 0.5) / n_bins,
    }


def make_global_rows(global_counts: dict[tuple[str, str], int]) -> list[dict[str, object]]:
    rows = []
    for aa in AA_ORDER:
        family = CODONS_BY_AA[aa]
        family_total = sum(global_counts.get((aa, codon), 0) for codon in family)
        for codon in family:
            count = global_counts.get((aa, codon), 0)
            rows.append(
                {
                    "reference_amino_acid": aa,
                    "amino_acid_name": AA_NAMES.get(aa, aa),
                    "reference_codon": codon,
                    "n_codons": count,
                    "family_total_codons": family_total,
                    "usage_fraction": count / family_total if family_total else None,
                }
            )
    return rows


def make_position_rows(
    bin_counts: dict[tuple[str, int, str], int],
    *,
    n_bins: int,
) -> list[dict[str, object]]:
    rows = []
    for aa in AA_ORDER:
        family = CODONS_BY_AA[aa]
        for bin_id in range(n_bins):
            family_total = sum(bin_counts.get((aa, bin_id, codon), 0) for codon in family)
            for codon in family:
                count = bin_counts.get((aa, bin_id, codon), 0)
                rows.append(
                    {
                        "reference_amino_acid": aa,
                        "amino_acid_name": AA_NAMES.get(aa, aa),
                        "reference_codon": codon,
                        **bin_columns(bin_id, n_bins),
                        "n_codons": count,
                        "family_bin_total_codons": family_total,
                        "usage_fraction": count / family_total if family_total else None,
                    }
                )
    return rows


def format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_table(rows: list[dict[str, object]], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    for suffix, delimiter in [(".csv", ","), (".tsv", "\t")]:
        with stem.with_suffix(suffix).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: format_csv_value(row.get(key)) for key in columns})


def safe_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "unnamed"


def setup_axes(ax: plt.Axes, aa: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Relative CDS position (5-prime to 3-prime)")
    ax.set_ylabel("Codon usage fraction")
    ax.set_title(f"{AA_NAMES.get(aa, aa)} ({aa}) codon usage by CDS position", pad=8)


def plot_by_amino_acid(rows: list[dict[str, object]], *, out_dir: Path, dpi: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key: defaultdict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        fraction = row["usage_fraction"]
        if fraction in ("", None):
            continue
        by_key[(str(row["reference_amino_acid"]), str(row["reference_codon"]))].append(
            (float(row["bin_center"]), float(fraction))
        )

    n_plots = 0
    for aa in AA_ORDER:
        family = CODONS_BY_AA[aa]
        if not any((aa, codon) in by_key for codon in family):
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.2 if len(family) <= 6 else 4.8))
        colors = plt.get_cmap("tab10").colors
        for i, codon in enumerate(family):
            points = by_key.get((aa, codon), [])
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4.0,
                linewidth=1.5,
                color=colors[i % len(colors)],
                label=codon,
            )
        setup_axes(ax, aa)
        legend = ax.legend(title="Reference codon", frameon=False, loc="best")
        if legend is not None:
            legend._legend_box.align = "left"
        fig.tight_layout()
        stem = out_dir / f"{safe_filename(aa)}_{safe_filename(AA_NAMES.get(aa, aa))}"
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        n_plots += 1
    return n_plots


def write_readme(
    out_dir: Path,
    *,
    data_dir: Path,
    summary: dict[str, object],
    n_plots: int,
    plots_mode: str,
) -> None:
    text = f"""# Codon usage by CDS position

Input data:
- {data_dir}

Tables:
- {out_dir / "tables" / "global_codon_usage.csv"}
- {out_dir / "tables" / "position_bins_by_amino_acid.csv"}

Plots:
- {out_dir / "plots" / "by_amino_acid"}

Usage definition:
- Codons are grouped by amino-acid family.
- For each amino acid and position bin, usage_fraction = codon count / total synonymous-family codon count.
- Fractions sum to one across synonymous codons for bins with at least one codon from that amino-acid family.

Full-CDS filter:
- {summary["full_cds_filter"]}
- This avoids counting transcripts whose CDS was partially removed by left-truncation.

Plots mode: {plots_mode}
Rendered amino-acid plots: {n_plots}

Summary:
```json
{json.dumps(summary, indent=2, sort_keys=True)}
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = (args.data_dir or default_data_dir()).resolve()
    out_dir = (args.out_dir or default_out_dir(data_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = data_dir / "X.npy"
    metadata_path = data_dir / "metadata.json"
    schema_path = data_dir / "schema.json"
    if not x_path.exists():
        raise SystemExit(f"Missing X.npy: {x_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata.json: {metadata_path}")
    if not schema_path.exists():
        raise SystemExit(f"Missing schema.json: {schema_path}")

    print("Codon usage by CDS position", flush=True)
    print(f"Input data: {data_dir}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise SystemExit(f"Expected schema.json to contain an object: {schema_path}")
    metadata_obj = load_json(metadata_path)
    if not isinstance(metadata_obj, list):
        raise SystemExit(f"Expected metadata.json to contain a list: {metadata_path}")
    metadata = [dict(item) for item in metadata_obj]

    X = np.load(x_path, mmap_mode="r")
    if X.ndim != 3:
        raise SystemExit(f"Expected X.npy with shape (N, C, L), got {X.shape}")
    if len(metadata) != int(X.shape[0]):
        raise SystemExit(f"metadata length {len(metadata)} does not match X.shape[0] {X.shape[0]}")

    base_indices, base_letters, cds_index = resolve_channels(schema)
    if max(int(i) for i in base_indices) >= int(X.shape[1]) or cds_index >= int(X.shape[1]):
        raise SystemExit(
            f"Resolved channels exceed X channel dimension {X.shape[1]}: "
            f"bases={base_indices.tolist()}, cds={cds_index}"
        )

    global_counts, bin_counts, summary = analyze_usage(
        X,
        metadata,
        base_indices=base_indices,
        base_letters=base_letters,
        cds_index=cds_index,
        n_bins=args.n_bins,
        limit_transcripts=args.limit_transcripts,
    )
    summary.update(
        {
            "x_shape": [int(value) for value in X.shape],
            "base_channel_indices": [int(value) for value in base_indices],
            "base_channel_letters": list(base_letters),
            "cds_codon_start_channel_index": int(cds_index),
        }
    )

    table_dir = out_dir / "tables"
    global_rows = make_global_rows(global_counts)
    position_rows = make_position_rows(bin_counts, n_bins=args.n_bins)
    write_table(global_rows, table_dir / "global_codon_usage")
    write_table(position_rows, table_dir / "position_bins_by_amino_acid")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    n_plots = 0
    if args.plots != "none":
        n_plots = plot_by_amino_acid(position_rows, out_dir=out_dir / "plots" / "by_amino_acid", dpi=args.dpi)

    write_readme(out_dir, data_dir=data_dir, summary=summary, n_plots=n_plots, plots_mode=args.plots)
    print(f"Done. Wrote tables, plots, and README under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
