"""Universal eCLIP preprocessing orchestration."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pysam

from transcriptml import __version__
from transcriptml.progress import log_progress
from transcriptml.rbpnet.annotation import parse_gtf
from transcriptml.rbpnet.fasta import retain_fasta_transcripts, write_transcript_fasta
from transcriptml.rbpnet.serialization import write_exons, write_json, write_metadata, write_regions
from transcriptml.rbpnet.signals import ExonBinIndex, create_signal_store, extract_bam_to_store, write_ip_pooled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """One named BAM sample and its experimental role."""

    name: str
    path: Path
    role: str


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for canonical transcript-space eCLIP preprocessing."""

    genome_fasta: Path
    gtf: Path
    sminput: Sample
    ips: tuple[Sample, ...]
    output_dir: Path
    read1_rna_strand: str = "opposite"
    min_mapq: int = 1
    exclude_duplicates: bool = True
    overwrite: bool = False
    progress: bool = True


_OUTPUT_FILES = (
    "transcripts.tsv", "exons.tsv.gz", "regions.tsv.gz", "transcripts.fa",
    "transcripts.fa.fai", "signals.h5", "qc.json", "manifest.json",
)


def _prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    conflicts = [path / name for name in _OUTPUT_FILES if (path / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            f"output files already exist in {path}; choose a new directory or pass --overwrite"
        )
    for conflict in conflicts:
        conflict.unlink()


def _input_record(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preprocess_eclip(config: PipelineConfig) -> dict:
    """Create a reusable canonical transcript-space eCLIP experiment.

    The HDF5 track is concatenated transcript space and stays lazy on read.
    This stage deliberately performs no peak calling or region selection.
    """

    started = time.time()
    if not config.ips:
        raise ValueError("at least one IP BAM is required")
    if config.sminput.role != "sminput" or any(sample.role != "ip" for sample in config.ips):
        raise ValueError("sample roles must be one 'sminput' followed by one or more 'ip' samples")
    if config.read1_rna_strand not in {"opposite", "same", "unstranded"}:
        raise ValueError("read1_rna_strand must be opposite, same, or unstranded")
    if config.min_mapq < 0:
        raise ValueError("min_mapq must be non-negative")
    sample_names = [config.sminput.name] + [sample.name for sample in config.ips]
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("sample names must be unique")

    log_progress(f"rbpnet preprocess: prepare {config.output_dir}", enabled=config.progress)
    _prepare_output(config.output_dir, config.overwrite)
    all_transcripts = parse_gtf(config.gtf, progress=config.progress)
    transcripts, contig_filter_qc = retain_fasta_transcripts(config.genome_fasta, all_transcripts)
    skipped = contig_filter_qc["transcripts_skipped_missing_fasta_contig"]
    if skipped:
        missing_contigs = list(contig_filter_qc["gtf_contigs_missing_from_fasta"])
        missing = ", ".join(missing_contigs[:10])
        if len(missing_contigs) > 10:
            missing += f", ... ({len(missing_contigs)} contigs total)"
        logger.warning(
            "Skipping %d transcript(s) on GTF contigs absent from the FASTA: %s",
            skipped,
            missing,
        )

    sequence_qc = write_transcript_fasta(
        config.output_dir / "transcripts.fa",
        config.genome_fasta,
        transcripts,
        progress=config.progress,
    )
    write_exons(config.output_dir / "exons.tsv.gz", transcripts, progress=config.progress)
    write_regions(config.output_dir / "regions.tsv.gz", transcripts, progress=config.progress)

    samples = [config.sminput, *config.ips]
    index = ExonBinIndex(transcripts)
    sample_counts: list[np.ndarray] = []
    bam_qc: dict[str, dict] = {}
    log_progress("rbpnet preprocess: create base-resolution signal store", enabled=config.progress)
    with create_signal_store(
        config.output_dir / "signals.h5",
        transcripts,
        sample_names,
        [sample.role for sample in samples],
    ) as store:
        for row, sample in enumerate(samples):
            log_progress(
                f"rbpnet preprocess: process {sample.name} ({sample.role})",
                enabled=config.progress,
            )
            sample_qc, counts = extract_bam_to_store(
                sample.path,
                row,
                store["counts"],
                transcripts,
                index,
                config.read1_rna_strand,
                config.min_mapq,
                config.exclude_duplicates,
                temp_dir=config.output_dir,
                progress=config.progress,
            )
            sample_qc["bam"] = str(sample.path.resolve())
            sample_qc["role"] = sample.role
            sample_qc["retained_fraction_of_read1"] = (
                sample_qc["retained"] / sample_qc["read1_seen"] if sample_qc["read1_seen"] else 0.0
            )
            sample_qc["retained_fraction_of_passing_filters"] = (
                sample_qc["retained"] / sample_qc["passing_filters"]
                if sample_qc["passing_filters"] else 0.0
            )
            bam_qc[sample.name] = sample_qc
            sample_counts.append(counts)
        write_ip_pooled(store, list(range(1, len(samples))), progress=config.progress)

    sm_tpm = write_metadata(
        config.output_dir / "transcripts.tsv",
        transcripts,
        sample_names,
        sample_counts,
        0,
        progress=config.progress,
    )
    region_counts: dict[str, int] = {}
    for tx in transcripts:
        for region in tx.regions:
            region_counts[region.label] = region_counts.get(region.label, 0) + region.end - region.start
    qc = {
        "format_version": "1",
        "pipeline_version": __version__,
        "configuration": {
            "read1_rna_strand": config.read1_rna_strand,
            "min_mapq": config.min_mapq,
            "exclude_duplicates": config.exclude_duplicates,
            "assignment": (
                "unique strand-compatible mature transcript at read1 5-prime aligned base, "
                "with all aligned CIGAR segments compatible with selected exons and junctions"
            ),
        },
        "annotation": {
            **contig_filter_qc,
            "transcripts": len(transcripts),
            "genes": len({tx.gene_id for tx in transcripts}),
            "exons": sum(len(tx.exons) for tx in transcripts),
            "transcriptome_bases": sum(tx.length for tx in transcripts),
            "chromosomes": sorted({tx.chrom for tx in transcripts}),
            "region_bases": dict(sorted(region_counts.items())),
            "sminput_transcripts_nonzero": int(np.count_nonzero(sample_counts[0])),
            "sminput_transcripts_tpm_ge_1": int(np.count_nonzero(sm_tpm >= 1.0)),
        },
        "sequence": sequence_qc,
        "samples": bam_qc,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(config.output_dir / "qc.json", qc)
    manifest = {
        "format": "transcriptml-rbpnet-experiment",
        "format_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": (
            "all intervals are 0-based, half-open; sequences/tracks are transcript 5-prime to 3-prime"
        ),
        "inputs": {
            "genome_fasta": _input_record(config.genome_fasta),
            "gtf": _input_record(config.gtf),
            "sminput_bam": _input_record(config.sminput.path),
            "ip_bams": [_input_record(sample.path) for sample in config.ips],
        },
        "configuration": qc["configuration"],
        "samples": [
            {
                "name": sample.name,
                "role": sample.role,
                "effective_library_size": bam_qc[sample.name]["retained"],
                "effective_library_size_count_type": "retained_read1_5prime_events",
            }
            for sample in samples
        ],
        "derived_signals": {
            "ip_pooled": {
                "source_samples": [sample.name for sample in config.ips],
                "effective_library_size": sum(bam_qc[sample.name]["retained"] for sample in config.ips),
                "effective_library_size_count_type": "sum_of_source_effective_library_sizes",
            }
        },
        "normalization": {
            "cpm_formula": "window_count / effective_library_size * 1e6",
            "sample_denominator_field": "samples[].effective_library_size",
            "effective_library_size_definition": (
                "retained read1 5-prime events used to construct the transcript-space signal track"
            ),
            "pooled_ip_denominator": "sum of effective_library_size over IP source samples",
        },
        "files": {
            "metadata": "transcripts.tsv",
            "exon_mapping": "exons.tsv.gz",
            "regions": "regions.tsv.gz",
            "sequences": "transcripts.fa",
            "signals": "signals.h5",
            "qc": "qc.json",
        },
        "software": {
            "transcriptml": __version__,
            "python": platform.python_version(),
            "pysam": pysam.__version__,
            "h5py": h5py.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)
    log_progress(
        f"rbpnet preprocess: complete in {time.time() - started:.1f}s",
        enabled=config.progress,
    )
    return qc


# A familiar alias for callers migrating from the standalone implementation.
run_pipeline = preprocess_eclip
