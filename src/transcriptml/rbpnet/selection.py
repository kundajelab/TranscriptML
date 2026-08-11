"""Explicit, versioned region selection over descriptive eCLIP windows."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import binom, poisson

from transcriptml.progress import ProgressReporter, log_progress
from transcriptml.rbpnet.experiment import ProcessedECLIPDataset
from transcriptml.rbpnet.windows import REGION_TYPES, summarize_regions

SELECTION_STRATEGIES = ("original_rbpnet", "yeo_2026", "peak_gray_negative")


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for selecting eligible experimental loci.

    Defaults for ``original_rbpnet`` reproduce the published Horlacher et al.
    candidate rules. Defaults for the other two strategies are transparent
    starting points and should be reviewed for each assay.
    """

    processed_dir: Path
    windows: Path
    output_prefix: Path
    strategy: str
    overwrite: bool = False
    progress: bool = True
    batch_size: int = 10_000
    # Published v1 selector.
    original_min_pvalue: float = 0.01
    original_min_count: int = 8
    original_min_height: int = 2
    original_advance: int = 50
    # Broad measured-window selector.
    min_total_count: int = 8
    min_sminput_count: int = 1
    min_ip_count: int = 1
    min_sminput_tpm: float = 0.0
    replicate_mode: str = "combined"
    # Peak / gray / confident-negative selector.
    peak_fdr: float = 0.05
    peak_min_log2_ratio: float = 1.0
    negative_fdr: float = 0.05
    negative_max_log2_ratio: float = -0.5
    stitch_gap: int = 0


@dataclass(frozen=True)
class SelectionManifest:
    """Loaded selection table and its versioned provenance metadata."""

    path: Path
    table: pa.Table
    metadata: dict

    @property
    def rows(self) -> list[dict]:
        return self.table.to_pylist()


def _resolve_parquet(path: Path) -> Path:
    if path.suffix == ".parquet":
        return path
    candidate = Path(str(path) + ".parquet")
    if candidate.is_file():
        return candidate
    raise ValueError("selection currently requires the scanner's Parquet table; pass its .parquet path or prefix")


def _scan_metadata(path: Path) -> dict:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"transcriptml_rbpnet_window_scan") or metadata.get(b"rbpnet_window_scan")
    if raw is None:
        raise ValueError(f"window Parquet lacks TranscriptML/RBPNet scan metadata: {path}")
    return json.loads(raw.decode())


def _iter_window_rows(path: Path, *, batch_size: int) -> Iterator[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _stable_example_id(
    strategy: str,
    transcript_id: str,
    start: int,
    end: int,
    state: str,
    replicate_id: str,
) -> str:
    payload = "\x1f".join(
        ("selection-v1", strategy, transcript_id, str(start), str(end), state, replicate_id)
    )
    return "rbp_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _manifest_schema(ds: ProcessedECLIPDataset, metadata: dict[bytes, bytes]) -> pa.Schema:
    fields = [
        pa.field("example_id", pa.string()),
        pa.field("gene_id", pa.string()),
        pa.field("transcript_id", pa.string()),
        pa.field("chromosome", pa.string()),
        pa.field("strand", pa.string()),
        pa.field("transcript_anchor", pa.int64()),
        pa.field("selection_start", pa.int64()),
        pa.field("selection_end", pa.int64()),
        pa.field("selection_length", pa.int64()),
        pa.field("region_type", pa.string()),
    ]
    for region_type in REGION_TYPES:
        fields.extend([
            pa.field(f"region_{region_type}_nt", pa.int64()),
            pa.field(f"region_{region_type}_fraction", pa.float64()),
        ])
    fields.extend([
        pa.field("genomic_blocks", pa.string()),
        pa.field("selection_strategy", pa.string()),
        pa.field("selection_state", pa.string()),
        pa.field("replicate_id", pa.string()),
        pa.field("source_window_count", pa.int64()),
        pa.field("sminput_tpm", pa.float64()),
    ])
    fields.extend(pa.field(f"{sample.name}_count", pa.int64()) for sample in ds.samples)
    fields.extend([
        pa.field("ip_pooled_count", pa.int64()),
    ])
    fields.extend(pa.field(f"{sample.name}_cpm", pa.float64()) for sample in ds.samples)
    fields.extend([
        pa.field("ip_pooled_cpm", pa.float64()),
        pa.field("total_ip_sminput_count", pa.int64()),
        pa.field("log2_ip_pooled_vs_sminput", pa.float64()),
    ])
    fields.extend(pa.field(f"max_{sample.name}_5pend", pa.int64()) for sample in ds.samples)
    fields.extend([
        pa.field("max_ip_pooled_5pend", pa.int64()),
        pa.field("selection_pvalue", pa.float64()),
        pa.field("selection_qvalue", pa.float64()),
        pa.field("source_min_enrichment_pvalue", pa.float64()),
        pa.field("source_min_enrichment_qvalue", pa.float64()),
        pa.field("source_min_depletion_pvalue", pa.float64()),
        pa.field("source_min_depletion_qvalue", pa.float64()),
        pa.field("group_gene_id", pa.string()),
        pa.field("group_transcript_id", pa.string()),
        pa.field("group_chromosome", pa.string()),
    ])
    return pa.schema(fields, metadata=metadata)


def _format_blocks(ds: ProcessedECLIPDataset, transcript_id: str, start: int, end: int) -> str:
    return ";".join(
        f"{block.chromosome}:{block.start}-{block.end}"
        for block in ds.get_genomic_blocks(transcript_id, start, end)
    )


def _base_manifest_row(
    ds: ProcessedECLIPDataset,
    source: dict,
    *,
    strategy: str,
    state: str,
    replicate_id: str = "",
    source_window_count: int = 1,
    anchor: int | None = None,
    selection_pvalue: float = math.nan,
    selection_qvalue: float = math.nan,
    enrichment_pvalue: float = math.nan,
    enrichment_qvalue: float = math.nan,
    depletion_pvalue: float = math.nan,
    depletion_qvalue: float = math.nan,
) -> dict:
    start = int(source["tx_start"])
    end = int(source["tx_end"])
    tx = ds.get_transcript(source["transcript_id"])
    anchor = start + (end - start) // 2 if anchor is None else int(anchor)
    row = {
        "example_id": _stable_example_id(strategy, tx.transcript_id, start, end, state, replicate_id),
        "gene_id": tx.gene_id,
        "transcript_id": tx.transcript_id,
        "chromosome": tx.chromosome,
        "strand": tx.strand,
        "transcript_anchor": anchor,
        "selection_start": start,
        "selection_end": end,
        "selection_length": end - start,
        "region_type": source["region_type"],
        "genomic_blocks": source["genomic_blocks"],
        "selection_strategy": strategy,
        "selection_state": state,
        "replicate_id": replicate_id,
        "source_window_count": source_window_count,
        "sminput_tpm": float(source["sminput_tpm"]),
        "ip_pooled_count": int(source["ip_pooled_count"]),
        "ip_pooled_cpm": float(source["ip_pooled_cpm"]),
        "total_ip_sminput_count": int(source["total_ip_sminput_count"]),
        "log2_ip_pooled_vs_sminput": float(source["log2_ip_pooled_vs_sminput"]),
        "max_ip_pooled_5pend": int(source["max_ip_pooled_5pend"]),
        "selection_pvalue": float(selection_pvalue),
        "selection_qvalue": float(selection_qvalue),
        "source_min_enrichment_pvalue": float(enrichment_pvalue),
        "source_min_enrichment_qvalue": float(enrichment_qvalue),
        "source_min_depletion_pvalue": float(depletion_pvalue),
        "source_min_depletion_qvalue": float(depletion_qvalue),
        "group_gene_id": tx.gene_id,
        "group_transcript_id": tx.transcript_id,
        "group_chromosome": tx.chromosome,
    }
    for region_type in REGION_TYPES:
        row[f"region_{region_type}_nt"] = int(source[f"region_{region_type}_nt"])
        row[f"region_{region_type}_fraction"] = float(source[f"region_{region_type}_fraction"])
    for sample in ds.samples:
        row[f"{sample.name}_count"] = int(source[f"{sample.name}_count"])
        row[f"{sample.name}_cpm"] = float(source[f"{sample.name}_cpm"])
        row[f"max_{sample.name}_5pend"] = int(source[f"max_{sample.name}_5pend"])
    return row


def _interval_source(
    ds: ProcessedECLIPDataset,
    transcript_id: str,
    start: int,
    end: int,
    *,
    pseudocount: float,
) -> dict:
    tx = ds.get_transcript(transcript_id)
    profiles = ds.get_profiles(transcript_id, start, end)
    pooled = ds.get_pooled_ip_profile(transcript_id, start, end)
    counts = profiles.sum(axis=1, dtype=np.uint64)
    pooled_count = int(pooled.sum(dtype=np.uint64))
    denominators = {sample.name: int(sample.effective_library_size) for sample in ds.samples}
    pooled_denominator = ds.pooled_ip_effective_library_size
    cpms = {
        sample.name: float(counts[i]) / denominators[sample.name] * 1_000_000.0
        for i, sample in enumerate(ds.samples)
    }
    pooled_cpm = pooled_count / pooled_denominator * 1_000_000.0
    sminput = ds.sminput_sample.name
    sminput_index = ds.sample_names.index(sminput)
    region_type, region_counts, region_fractions = summarize_regions(tx.regions, start, end)
    source = {
        "transcript_id": transcript_id,
        "tx_start": start,
        "tx_end": end,
        "region_type": region_type,
        "genomic_blocks": _format_blocks(ds, transcript_id, start, end),
        "sminput_tpm": tx.sminput_tpm,
        "ip_pooled_count": pooled_count,
        "ip_pooled_cpm": pooled_cpm,
        "total_ip_sminput_count": pooled_count + int(counts[sminput_index]),
        "log2_ip_pooled_vs_sminput": math.log2(
            (pooled_cpm + pseudocount) / (cpms[sminput] + pseudocount)
        ),
        "max_ip_pooled_5pend": int(pooled.max(initial=0)),
    }
    for region in REGION_TYPES:
        source[f"region_{region}_nt"] = region_counts[region]
        source[f"region_{region}_fraction"] = region_fractions[region]
    for i, sample in enumerate(ds.samples):
        source[f"{sample.name}_count"] = int(counts[i])
        source[f"{sample.name}_cpm"] = cpms[sample.name]
        source[f"max_{sample.name}_5pend"] = int(profiles[i].max(initial=0))
    return source


def _bh_adjust(pvalues: np.ndarray, tested: np.ndarray) -> np.ndarray:
    qvalues = np.ones(pvalues.shape, dtype=np.float64)
    indices = np.nonzero(tested)[0]
    if indices.size == 0:
        return qvalues
    order = indices[np.argsort(pvalues[indices], kind="stable")]
    ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues[order] = np.minimum(ranked, 1.0)
    return qvalues


def _original_rows(
    config: SelectionConfig,
    ds: ProcessedECLIPDataset,
    windows_path: Path,
    scan_metadata: dict,
) -> Iterator[dict]:
    if int(scan_metadata.get("window_size", -1)) != 100 or int(scan_metadata.get("stride", -1)) != 1:
        raise ValueError(
            "original_rbpnet requires a 100-nt, stride-1 scan; rerun scan-windows "
            "with --window-size 100 --stride 1"
        )
    if not bool(scan_metadata.get("omit_incomplete_terminal_windows", False)):
        raise ValueError("original_rbpnet requires incomplete terminal windows to be omitted")
    current_tx = None
    mu = 0.0
    next_start = 0
    reporter = ProgressReporter(
        "rbpnet select-regions: test v1 windows",
        total=pq.ParquetFile(windows_path).metadata.num_rows,
        unit="windows",
        enabled=config.progress,
    )
    try:
        for row in _iter_window_rows(windows_path, batch_size=config.batch_size):
            reporter.update()
            tx_id = row["transcript_id"]
            if tx_id != current_tx:
                tx = ds.get_transcript(tx_id)
                transcript_count = int(ds.get_pooled_ip_profile(tx_id, 0, tx.length).sum(dtype=np.uint64))
                mu = transcript_count / tx.length * int(row["window_length"])
                current_tx = tx_id
                next_start = 0
            start = int(row["tx_start"])
            if start < next_start:
                continue
            count = int(row["ip_pooled_count"])
            height = int(row["max_ip_pooled_5pend"])
            pvalue = float(poisson.sf(count - 1, mu))
            if (
                pvalue < config.original_min_pvalue
                and count >= config.original_min_count
                and height >= config.original_min_height
            ):
                yield _base_manifest_row(
                    ds,
                    row,
                    strategy="original_rbpnet",
                    state="candidate",
                    selection_pvalue=pvalue,
                )
                next_start = start + config.original_advance
    finally:
        reporter.close()


def _yeo_rows(
    config: SelectionConfig,
    ds: ProcessedECLIPDataset,
    windows_path: Path,
) -> Iterator[dict]:
    input_name = ds.sminput_sample.name
    reporter = ProgressReporter(
        "rbpnet select-regions: filter measured windows",
        total=pq.ParquetFile(windows_path).metadata.num_rows,
        unit="windows",
        enabled=config.progress,
    )
    try:
        for row in _iter_window_rows(windows_path, batch_size=config.batch_size):
            reporter.update()
            if float(row["sminput_tpm"]) < config.min_sminput_tpm:
                continue
            input_count = int(row[f"{input_name}_count"])
            if config.replicate_mode == "combined":
                ip_count = int(row["ip_pooled_count"])
                if (
                    input_count + ip_count >= config.min_total_count
                    and input_count >= config.min_sminput_count
                    and ip_count >= config.min_ip_count
                ):
                    yield _base_manifest_row(
                        ds,
                        row,
                        strategy="yeo_2026",
                        state="measured",
                        replicate_id="",
                    )
            else:
                for sample in ds.ip_samples:
                    ip_count = int(row[f"{sample.name}_count"])
                    if (
                        input_count + ip_count >= config.min_total_count
                        and input_count >= config.min_sminput_count
                        and ip_count >= config.min_ip_count
                    ):
                        yield _base_manifest_row(
                            ds,
                            row,
                            strategy="yeo_2026",
                            state="measured",
                            replicate_id=sample.name,
                        )
    finally:
        reporter.close()


def _peak_statistics(
    config: SelectionConfig,
    ds: ProcessedECLIPDataset,
    windows_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_name = ds.sminput_sample.name
    log_progress("rbpnet select-regions: calculate window statistics", enabled=config.progress)
    table = pq.read_table(
        windows_path,
        columns=[
            f"{input_name}_count", "ip_pooled_count", "total_ip_sminput_count",
            "sminput_tpm",
        ],
    )
    input_counts = table[f"{input_name}_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    ip_counts = table["ip_pooled_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    totals = table["total_ip_sminput_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    tpm = table["sminput_tpm"].to_numpy(zero_copy_only=False).astype(np.float64)
    adequate = (
        (totals >= config.min_total_count)
        & (input_counts >= config.min_sminput_count)
        & (ip_counts >= config.min_ip_count)
        & (tpm >= config.min_sminput_tpm)
    )
    input_size = int(ds.sminput_sample.effective_library_size)
    ip_size = ds.pooled_ip_effective_library_size
    null_ip_probability = ip_size / (ip_size + input_size)
    enrichment_p = np.ones(len(totals), dtype=np.float64)
    depletion_p = np.ones(len(totals), dtype=np.float64)
    enrichment_p[adequate] = binom.sf(
        ip_counts[adequate] - 1, totals[adequate], null_ip_probability
    )
    depletion_p[adequate] = binom.cdf(
        ip_counts[adequate], totals[adequate], null_ip_probability
    )
    log_progress(
        f"rbpnet select-regions: {int(adequate.sum()):,}/{len(adequate):,} windows adequately measured",
        enabled=config.progress,
    )
    return (
        adequate,
        enrichment_p,
        _bh_adjust(enrichment_p, adequate),
        depletion_p,
        _bh_adjust(depletion_p, adequate),
    )


def _peak_gray_negative_rows(
    config: SelectionConfig,
    ds: ProcessedECLIPDataset,
    windows_path: Path,
    scan_metadata: dict,
) -> Iterator[dict]:
    adequate, enrichment_p, enrichment_q, depletion_p, depletion_q = _peak_statistics(
        config, ds, windows_path
    )
    current: dict | None = None
    reporter = ProgressReporter(
        "rbpnet select-regions: classify and stitch windows",
        total=len(adequate),
        unit="windows",
        enabled=config.progress,
    )

    def flush_current() -> dict | None:
        nonlocal current
        if current is None:
            return None
        source = _interval_source(
            ds,
            current["transcript_id"],
            current["start"],
            current["end"],
            pseudocount=float(scan_metadata["pseudocount_cpm"]),
        )
        if current["state"] == "peak":
            pooled = ds.get_pooled_ip_profile(
                current["transcript_id"], current["start"], current["end"]
            )
            maximum = pooled.max(initial=0)
            candidates = np.flatnonzero(pooled == maximum)
            center = (len(pooled) - 1) / 2
            local_anchor = min(candidates.tolist(), key=lambda x: (abs(x - center), x))
            anchor = current["start"] + int(local_anchor)
            selection_pvalue = current["min_enrichment_p"]
            selection_qvalue = current["min_enrichment_q"]
        else:
            anchor = current["start"] + (current["end"] - current["start"]) // 2
            if current["state"] == "confident_negative":
                selection_pvalue = current["min_depletion_p"]
                selection_qvalue = current["min_depletion_q"]
            else:
                selection_pvalue = math.nan
                selection_qvalue = math.nan
        result = _base_manifest_row(
            ds,
            source,
            strategy="peak_gray_negative",
            state=current["state"],
            source_window_count=current["source_window_count"],
            anchor=anchor,
            selection_pvalue=selection_pvalue,
            selection_qvalue=selection_qvalue,
            enrichment_pvalue=current["min_enrichment_p"],
            enrichment_qvalue=current["min_enrichment_q"],
            depletion_pvalue=current["min_depletion_p"],
            depletion_qvalue=current["min_depletion_q"],
        )
        current = None
        return result

    try:
        for index, row in enumerate(_iter_window_rows(windows_path, batch_size=config.batch_size)):
            reporter.update()
            if not adequate[index]:
                flushed = flush_current()
                if flushed is not None:
                    yield flushed
                continue
            ratio = float(row["log2_ip_pooled_vs_sminput"])
            if enrichment_q[index] <= config.peak_fdr and ratio >= config.peak_min_log2_ratio:
                state = "peak"
            elif depletion_q[index] <= config.negative_fdr and ratio <= config.negative_max_log2_ratio:
                state = "confident_negative"
            else:
                state = "gray"
            compatible = (
                current is not None
                and current["transcript_id"] == row["transcript_id"]
                and current["state"] == state
                and current["region_type"] == row["region_type"]
                and int(row["tx_start"]) <= current["end"] + config.stitch_gap
            )
            if not compatible:
                flushed = flush_current()
                if flushed is not None:
                    yield flushed
                current = {
                    "transcript_id": row["transcript_id"],
                    "start": int(row["tx_start"]),
                    "end": int(row["tx_end"]),
                    "state": state,
                    "region_type": row["region_type"],
                    "source_window_count": 1,
                    "min_enrichment_p": float(enrichment_p[index]),
                    "min_enrichment_q": float(enrichment_q[index]),
                    "min_depletion_p": float(depletion_p[index]),
                    "min_depletion_q": float(depletion_q[index]),
                }
            else:
                current["end"] = max(current["end"], int(row["tx_end"]))
                current["source_window_count"] += 1
                current["min_enrichment_p"] = min(current["min_enrichment_p"], float(enrichment_p[index]))
                current["min_enrichment_q"] = min(current["min_enrichment_q"], float(enrichment_q[index]))
                current["min_depletion_p"] = min(current["min_depletion_p"], float(depletion_p[index]))
                current["min_depletion_q"] = min(current["min_depletion_q"], float(depletion_q[index]))
        flushed = flush_current()
        if flushed is not None:
            yield flushed
    finally:
        reporter.close()


def _validate_config(config: SelectionConfig) -> None:
    if config.strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"unknown selection strategy {config.strategy!r}")
    if config.batch_size <= 0 or config.original_advance <= 0:
        raise ValueError("batch_size and original_advance must be positive")
    for name in ("original_min_pvalue", "peak_fdr", "negative_fdr"):
        value = float(getattr(config, name))
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if min(config.original_min_count, config.original_min_height, config.min_total_count,
           config.min_sminput_count, config.min_ip_count, config.stitch_gap) < 0:
        raise ValueError("count thresholds and stitch_gap must be non-negative")
    if config.min_sminput_tpm < 0:
        raise ValueError("min_sminput_tpm must be non-negative")
    if config.replicate_mode not in {"combined", "per_ip"}:
        raise ValueError("replicate_mode must be combined or per_ip")


def _validate_scan_dataset(
    ds: ProcessedECLIPDataset,
    windows_path: Path,
    scan_metadata: dict,
) -> None:
    expected_sizes = {
        sample.name: int(sample.effective_library_size)
        for sample in ds.samples
    }
    observed_sizes = {
        str(name): int(value)
        for name, value in scan_metadata.get("effective_library_sizes", {}).items()
    }
    if observed_sizes != expected_sizes:
        raise ValueError(
            "window scan effective library sizes do not match the processed experiment"
        )
    if int(scan_metadata.get("ip_pooled_effective_library_size", -1)) != ds.pooled_ip_effective_library_size:
        raise ValueError("window scan pooled-IP library size does not match the processed experiment")
    required_columns = {
        "transcript_id", "tx_start", "tx_end", "window_length", "region_type",
        "sminput_tpm", "ip_pooled_count", "ip_pooled_cpm",
        "total_ip_sminput_count", "log2_ip_pooled_vs_sminput",
        "max_ip_pooled_5pend", "genomic_blocks",
    }
    for sample in ds.samples:
        required_columns.update({
            f"{sample.name}_count", f"{sample.name}_cpm", f"max_{sample.name}_5pend"
        })
    missing = sorted(required_columns - set(pq.read_schema(windows_path).names))
    if missing:
        raise ValueError(f"window table lacks columns required by this experiment: {', '.join(missing)}")


def select_regions(config: SelectionConfig) -> dict:
    """Select biological loci and write a versioned lightweight manifest."""

    _validate_config(config)
    windows_path = _resolve_parquet(config.windows)
    scan_metadata = _scan_metadata(windows_path)
    prefix = str(config.output_prefix)
    if prefix.endswith((".parquet", ".tsv", ".tsv.gz", ".selection.json")):
        raise ValueError("output_prefix must not include a table or metadata suffix")
    parquet_path = Path(prefix + ".parquet")
    tsv_path = Path(prefix + ".tsv.gz")
    sidecar_path = Path(prefix + ".selection.json")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts = [path for path in (parquet_path, tsv_path, sidecar_path) if path.exists()]
    if conflicts and not config.overwrite:
        raise FileExistsError(f"selection output already exists ({conflicts[0]}); pass --overwrite")

    log_progress(f"rbpnet select-regions: {config.strategy}", enabled=config.progress)
    with ProcessedECLIPDataset(config.processed_dir) as ds:
        if any(sample.effective_library_size is None or sample.effective_library_size <= 0 for sample in ds.samples):
            raise ValueError("all samples need positive effective_library_size values for selection")
        _validate_scan_dataset(ds, windows_path, scan_metadata)
        provenance = {
            "format": "transcriptml-rbpnet-selection",
            "format_version": "1",
            "strategy": config.strategy,
            "source_processed_dir": str(config.processed_dir.resolve()),
            "source_windows": str(windows_path.resolve()),
            "window_scan": scan_metadata,
            "configuration": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in config.__dict__.items()
                if key not in {"processed_dir", "windows", "output_prefix", "progress"}
            },
            "statistical_notes": (
                "original_rbpnet uses a one-sided Poisson test against the transcript-level pooled-IP rate"
                if config.strategy == "original_rbpnet"
                else "peak_gray_negative uses exact conditional binomial tails and BH correction over adequately measured windows"
                if config.strategy == "peak_gray_negative"
                else "yeo_2026 applies coverage thresholds only and performs no peak test"
            ),
        }
        schema = _manifest_schema(
            ds,
            {b"transcriptml_rbpnet_selection": json.dumps(provenance, sort_keys=True).encode()},
        )
        if config.strategy == "original_rbpnet":
            rows: Iterable[dict] = _original_rows(config, ds, windows_path, scan_metadata)
        elif config.strategy == "yeo_2026":
            rows = _yeo_rows(config, ds, windows_path)
        else:
            rows = _peak_gray_negative_rows(config, ds, windows_path, scan_metadata)

        selected = 0
        state_counts: Counter[str] = Counter()
        transcript_ids: set[str] = set()
        batch: list[dict] = []
        reporter = ProgressReporter(
            "rbpnet select-regions: write manifest",
            total=None,
            unit="examples",
            enabled=config.progress,
        )
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

            for row in rows:
                batch.append(row)
                selected += 1
                state_counts[row["selection_state"]] += 1
                transcript_ids.add(row["transcript_id"])
                reporter.update()
                if len(batch) >= config.batch_size:
                    flush()
            flush()
        reporter.close()
        summary = {
            **provenance,
            "n_examples": selected,
            "n_transcripts": len(transcript_ids),
            "state_counts": dict(sorted(state_counts.items())),
            "parquet": str(parquet_path),
            "tsv": str(tsv_path),
        }
        sidecar_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_progress(
            f"rbpnet select-regions: wrote {selected:,} examples",
            enabled=config.progress,
        )
        return summary


def load_selection_manifest(path: str | Path) -> SelectionManifest:
    """Load and validate a version-1 Parquet selection manifest."""

    manifest_path = _resolve_parquet(Path(path))
    table = pq.read_table(manifest_path)
    raw = (table.schema.metadata or {}).get(b"transcriptml_rbpnet_selection")
    if raw is None:
        raise ValueError(f"selection manifest metadata is missing: {manifest_path}")
    metadata = json.loads(raw.decode())
    if metadata.get("format") != "transcriptml-rbpnet-selection" or str(metadata.get("format_version")) != "1":
        raise ValueError("unsupported RBPNet selection manifest format/version")
    required = {
        "example_id", "gene_id", "transcript_id", "chromosome", "strand",
        "transcript_anchor", "selection_start", "selection_end", "selection_strategy",
        "selection_state", "group_gene_id", "group_transcript_id", "group_chromosome",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"selection manifest lacks columns: {', '.join(missing)}")
    ids = table["example_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("selection manifest contains duplicate example_id values")
    return SelectionManifest(manifest_path, table, metadata)
