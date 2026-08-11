"""Interoperable metadata and coordinate-table serializers."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from transcriptml.rbpnet._progress import track
from transcriptml.rbpnet.coordinates import Transcript


def calculate_tpm(raw_counts: np.ndarray, transcripts: list[Transcript]) -> np.ndarray:
    """Calculate length-normalized TPM from retained transcript event counts."""

    lengths_kb = np.asarray([tx.length / 1000.0 for tx in transcripts], dtype=np.float64)
    rates = raw_counts.astype(np.float64) / lengths_kb
    denominator = rates.sum()
    return rates / denominator * 1_000_000.0 if denominator else np.zeros_like(rates)


def write_metadata(
    path: str | Path,
    transcripts: list[Transcript],
    sample_names: list[str],
    sample_counts: list[np.ndarray],
    sminput_index: int,
    *,
    progress: bool = True,
) -> np.ndarray:
    """Write the transcript metadata table and return SMInput TPM values."""

    tpm = calculate_tpm(sample_counts[sminput_index], transcripts)
    fields = [
        "transcript_id", "gene_id", "gene_name", "transcript_name", "transcript_type",
        "chrom", "strand", "transcript_length", "signal_offset", "sm_input_raw_count",
        "sm_input_tpm",
    ] + [f"{name}_raw_5p_count" for name in sample_names] + ["region_annotations"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, tx in enumerate(track(
            transcripts,
            "rbpnet preprocess: write transcript metadata",
            total=len(transcripts),
            unit="transcripts",
            enabled=progress,
        )):
            row = {
                "transcript_id": tx.transcript_id,
                "gene_id": tx.gene_id,
                "gene_name": tx.gene_name,
                "transcript_name": tx.transcript_name,
                "transcript_type": tx.transcript_type,
                "chrom": tx.chrom,
                "strand": tx.strand,
                "transcript_length": tx.length,
                "signal_offset": tx.offset,
                "sm_input_raw_count": int(sample_counts[sminput_index][index]),
                "sm_input_tpm": f"{tpm[index]:.8g}",
                "region_annotations": json.dumps(
                    [{"start": r.start, "end": r.end, "type": r.label} for r in tx.regions],
                    separators=(",", ":"),
                ),
            }
            for sample_name, counts in zip(sample_names, sample_counts):
                row[f"{sample_name}_raw_5p_count"] = int(counts[index])
            writer.writerow(row)
    return tpm


def write_exons(path: str | Path, transcripts: list[Transcript], *, progress: bool = True) -> None:
    """Write compressed transcript/genome exon mappings."""

    fields = [
        "transcript_id", "exon_index_5to3", "exon_number", "exon_id", "tx_start", "tx_end",
        "chrom", "genomic_start", "genomic_end", "strand",
    ]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for tx in track(
            transcripts,
            "rbpnet preprocess: write exon mappings",
            total=len(transcripts),
            unit="transcripts",
            enabled=progress,
        ):
            for index, exon in enumerate(tx.exons, 1):
                writer.writerow({
                    "transcript_id": tx.transcript_id,
                    "exon_index_5to3": index,
                    "exon_number": exon.exon_number,
                    "exon_id": exon.exon_id,
                    "tx_start": exon.tx_start,
                    "tx_end": exon.tx_end,
                    "chrom": exon.chrom,
                    "genomic_start": exon.start,
                    "genomic_end": exon.end,
                    "strand": exon.strand,
                })


def write_regions(path: str | Path, transcripts: list[Transcript], *, progress: bool = True) -> None:
    """Write compressed transcript region annotations."""

    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["transcript_id", "tx_start", "tx_end", "region_type"])
        for tx in track(
            transcripts,
            "rbpnet preprocess: write region annotations",
            total=len(transcripts),
            unit="transcripts",
            enabled=progress,
        ):
            for region in tx.regions:
                writer.writerow([tx.transcript_id, region.start, region.end, region.label])


def write_json(path: str | Path, value: dict) -> None:
    """Write deterministic indented JSON with a trailing newline."""

    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
