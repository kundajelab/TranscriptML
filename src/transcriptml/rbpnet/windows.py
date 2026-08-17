"""Descriptive configurable scanning over canonical transcript-space signals."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transcriptml.progress import ProgressReporter, log_progress
from transcriptml.rbpnet.experiment import ProcessedECLIPDataset, RegionRecord

REGION_TYPES = ("5putr", "cds", "3putr", "noncoding_exon", "intron")


@dataclass(frozen=True)
class WindowScanConfig:
    """Configuration for a descriptive transcript-window scan."""

    processed_dir: Path
    output_prefix: Path
    window_size: int = 100
    stride: int = 50
    min_sminput_tpm: float = 0.0
    pseudocount: float = 1.0
    omit_incomplete_terminal_windows: bool = True
    overwrite: bool = False
    batch_size: int = 10_000
    progress: bool = True


def generate_window_bounds(
    transcript_length: int,
    window_size: int,
    stride: int,
    omit_incomplete_terminal_windows: bool = True,
) -> Iterator[tuple[int, int]]:
    """Yield deterministic zero-based, half-open transcript windows."""

    if transcript_length < 0:
        raise ValueError("transcript length must be non-negative")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    for start in range(0, transcript_length, stride):
        end = start + window_size
        if end > transcript_length:
            if omit_incomplete_terminal_windows:
                break
            end = transcript_length
        if end > start:
            yield start, end


def calculate_gc_fraction(sequence: str) -> float:
    """Calculate GC bases divided by total length; ambiguous bases are non-GC."""

    if not sequence:
        return 0.0
    upper = sequence.upper()
    return (upper.count("G") + upper.count("C")) / len(upper)


def summarize_regions(
    regions: Iterable[RegionRecord], start: int, end: int
) -> tuple[str, dict[str, int], dict[str, float]]:
    """Summarize exact region overlap and label boundary-crossing windows mixed."""

    length = end - start
    if length <= 0:
        raise ValueError("window must have positive length")
    counts = {region_type: 0 for region_type in REGION_TYPES}
    for region in regions:
        if region.region_type not in counts:
            raise ValueError(f"unsupported region type in processed metadata: {region.region_type}")
        counts[region.region_type] += max(0, min(end, region.end) - max(start, region.start))
    covered = sum(counts.values())
    if covered != length:
        raise ValueError(f"region annotations cover {covered} of {length} bases for window {start}-{end}")
    present = [region_type for region_type, count in counts.items() if count]
    region_type = present[0] if len(present) == 1 else "mixed"
    return region_type, counts, {key: value / length for key, value in counts.items()}


def _window_schema(sample_names: tuple[str, ...], metadata: dict[bytes, bytes]) -> pa.Schema:
    fields = [
        pa.field("gene_id", pa.string()),
        pa.field("transcript_id", pa.string()),
        pa.field("chromosome", pa.string()),
        pa.field("strand", pa.string()),
        pa.field("tx_start", pa.int64()),
        pa.field("tx_end", pa.int64()),
        pa.field("window_length", pa.int64()),
        pa.field("region_type", pa.string()),
    ]
    for region_type in REGION_TYPES:
        fields.append(pa.field(f"region_{region_type}_nt", pa.int64()))
        fields.append(pa.field(f"region_{region_type}_fraction", pa.float64()))
    fields.extend([
        pa.field("gc_fraction", pa.float64()),
        pa.field("sminput_tpm", pa.float64()),
        pa.field("genomic_blocks", pa.string()),
    ])
    fields.extend(pa.field(f"{sample}_count", pa.int64()) for sample in sample_names)
    fields.append(pa.field("ip_pooled_count", pa.int64()))
    fields.extend(pa.field(f"{sample}_cpm", pa.float64()) for sample in sample_names)
    fields.append(pa.field("ip_pooled_cpm", pa.float64()))
    fields.extend([
        pa.field("total_ip_sminput_count", pa.int64()),
        pa.field("log2_ip_pooled_vs_sminput", pa.float64()),
    ])
    fields.extend(pa.field(f"max_{sample}_5pend", pa.int64()) for sample in sample_names)
    fields.append(pa.field("max_ip_pooled_5pend", pa.int64()))
    return pa.schema(fields, metadata=metadata)


def _format_blocks(ds: ProcessedECLIPDataset, tx_id: str, start: int, end: int) -> str:
    return ";".join(
        f"{block.chromosome}:{block.start}-{block.end}"
        for block in ds.get_genomic_blocks(tx_id, start, end)
    )


def _validate_config(config: WindowScanConfig) -> None:
    if config.window_size <= 0 or config.stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if config.min_sminput_tpm < 0:
        raise ValueError("min_sminput_tpm must be non-negative")
    if config.pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")


def scan_windows(config: WindowScanConfig) -> dict:
    """Write equivalent gzipped TSV and Parquet descriptive window tables."""

    _validate_config(config)
    prefix_text = str(config.output_prefix)
    if prefix_text.endswith((".tsv", ".tsv.gz", ".parquet", ".scan.json")):
        raise ValueError("output_prefix must not include a table or metadata suffix")
    tsv_path = Path(prefix_text + ".tsv.gz")
    parquet_path = Path(prefix_text + ".parquet")
    metadata_path = Path(prefix_text + ".scan.json")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts = [path for path in (tsv_path, parquet_path, metadata_path) if path.exists()]
    if conflicts and not config.overwrite:
        raise FileExistsError(f"window output already exists ({conflicts[0]}); pass --overwrite to replace it")

    log_progress(f"rbpnet scan-windows: open {config.processed_dir}", enabled=config.progress)
    with ProcessedECLIPDataset(config.processed_dir) as ds:
        if not ds.ip_samples:
            raise ValueError("processed dataset contains no IP samples")
        if "ip_pooled" in ds.sample_names:
            raise ValueError("sample name 'ip_pooled' is reserved for the derived pooled signal")
        missing_sizes = [sample.name for sample in ds.samples if sample.effective_library_size is None]
        if missing_sizes:
            raise ValueError(
                "manifest lacks effective_library_size for sample(s) "
                f"{', '.join(missing_sizes)}; rerun preprocessing with the current package"
            )
        zero_sizes = [sample.name for sample in ds.samples if int(sample.effective_library_size) <= 0]
        if zero_sizes:
            raise ValueError(f"effective_library_size must be positive for CPM: {', '.join(zero_sizes)}")
        denominators = {sample.name: int(sample.effective_library_size) for sample in ds.samples}
        pooled_denominator = sum(denominators[sample.name] for sample in ds.ip_samples)
        derived = ds.manifest.get("derived_signals", {}).get("ip_pooled", {})
        if "effective_library_size" in derived and int(derived["effective_library_size"]) != pooled_denominator:
            raise ValueError("manifest pooled-IP denominator disagrees with summed IP denominators")

        scan_metadata = {
            "format": "transcriptml-rbpnet-window-scan",
            "format_version": "1",
            "source_processed_dir": str(config.processed_dir.resolve()),
            "coordinate_space": ds.coordinate_space,
            "window_size": config.window_size,
            "stride": config.stride,
            "min_sminput_tpm": config.min_sminput_tpm,
            "pseudocount_cpm": config.pseudocount,
            "omit_incomplete_terminal_windows": config.omit_incomplete_terminal_windows,
            "effective_library_sizes": denominators,
            "ip_pooled_effective_library_size": pooled_denominator,
            "log_ratio_formula": "log2((ip_pooled_cpm+pseudocount)/(sminput_cpm+pseudocount))",
        }
        arrow_metadata = {
            b"transcriptml_rbpnet_window_scan": json.dumps(scan_metadata, sort_keys=True).encode()
        }
        schema = _window_schema(ds.sample_names, arrow_metadata)
        summary = {
            **scan_metadata,
            "transcripts_total": len(ds.transcripts),
            "transcripts_passing_sminput_tpm": 0,
            "transcripts_scanned": 0,
            "windows": 0,
            "region_type_windows": Counter(),
            "tsv": str(tsv_path),
            "parquet": str(parquet_path),
        }
        batch: list[dict] = []
        with gzip.open(tsv_path, "wt", newline="") as tsv_handle, pq.ParquetWriter(
            parquet_path, schema, compression="zstd"
        ) as parquet_writer:
            tsv_writer = csv.DictWriter(
                tsv_handle, fieldnames=schema.names, delimiter="\t", lineterminator="\n"
            )
            tsv_writer.writeheader()

            def flush() -> None:
                if not batch:
                    return
                tsv_writer.writerows(batch)
                parquet_writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()

            reporter = ProgressReporter(
                "rbpnet scan-windows: scan transcripts",
                total=len(ds.transcripts),
                unit="transcripts",
                enabled=config.progress,
            )
            for tx in ds.transcripts:
                if tx.sminput_tpm < config.min_sminput_tpm:
                    reporter.update()
                    continue
                summary["transcripts_passing_sminput_tpm"] += 1
                emitted = False
                sequence = ds.get_sequence(tx.transcript_id, 0, tx.length)
                profiles = ds.get_profiles(tx.transcript_id, 0, tx.length)
                pooled_profile = ds.get_pooled_ip_profile(tx.transcript_id, 0, tx.length)
                # Prefix sums make count aggregation O(1) per window, including
                # stride-1 scans used by the published v1 selector.
                profile_prefix = np.pad(
                    profiles.astype(np.uint64).cumsum(axis=1), ((0, 0), (1, 0))
                )
                pooled_prefix = np.pad(pooled_profile.astype(np.uint64).cumsum(), (1, 0))
                gc = np.fromiter((base.upper() in {"G", "C"} for base in sequence), dtype=np.uint8)
                gc_prefix = np.pad(gc.astype(np.uint64).cumsum(), (1, 0))
                for start, end in generate_window_bounds(
                    tx.length,
                    config.window_size,
                    config.stride,
                    config.omit_incomplete_terminal_windows,
                ):
                    emitted = True
                    window_profiles = profiles[:, start:end]
                    window_pooled = pooled_profile[start:end]
                    counts = profile_prefix[:, end] - profile_prefix[:, start]
                    pooled_count = int(pooled_prefix[end] - pooled_prefix[start])
                    region_type, region_counts, region_fractions = summarize_regions(tx.regions, start, end)
                    cpms = {
                        sample.name: float(counts[index]) / denominators[sample.name] * 1_000_000.0
                        for index, sample in enumerate(ds.samples)
                    }
                    pooled_cpm = pooled_count / pooled_denominator * 1_000_000.0
                    sminput_name = ds.sminput_sample.name
                    sminput_index = ds.sample_names.index(sminput_name)
                    row = {
                        "gene_id": tx.gene_id,
                        "transcript_id": tx.transcript_id,
                        "chromosome": tx.chromosome,
                        "strand": tx.strand,
                        "tx_start": start,
                        "tx_end": end,
                        "window_length": end - start,
                        "region_type": region_type,
                        "gc_fraction": float(gc_prefix[end] - gc_prefix[start]) / (end - start),
                        "sminput_tpm": tx.sminput_tpm,
                        "genomic_blocks": _format_blocks(ds, tx.transcript_id, start, end),
                        "ip_pooled_count": pooled_count,
                        "ip_pooled_cpm": pooled_cpm,
                        "total_ip_sminput_count": pooled_count + int(counts[sminput_index]),
                        "log2_ip_pooled_vs_sminput": math.log2(
                            (pooled_cpm + config.pseudocount)
                            / (cpms[sminput_name] + config.pseudocount)
                        ),
                        "max_ip_pooled_5pend": int(window_pooled.max(initial=0)),
                    }
                    for label in REGION_TYPES:
                        row[f"region_{label}_nt"] = region_counts[label]
                        row[f"region_{label}_fraction"] = region_fractions[label]
                    for index, sample in enumerate(ds.samples):
                        row[f"{sample.name}_count"] = int(counts[index])
                        row[f"{sample.name}_cpm"] = cpms[sample.name]
                        row[f"max_{sample.name}_5pend"] = int(window_profiles[index].max(initial=0))
                    batch.append(row)
                    summary["windows"] += 1
                    summary["region_type_windows"][region_type] += 1
                    if len(batch) >= config.batch_size:
                        flush()
                if emitted:
                    summary["transcripts_scanned"] += 1
                reporter.update(extra=f"{summary['windows']:,} windows")
            reporter.close(extra=f"{summary['windows']:,} windows")
            flush()
        summary["region_type_windows"] = dict(sorted(summary["region_type_windows"].items()))
        metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_progress(
            f"rbpnet scan-windows: wrote {summary['windows']:,} windows",
            enabled=config.progress,
        )
        return summary
