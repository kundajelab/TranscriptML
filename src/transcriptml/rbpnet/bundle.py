"""Materialize selected eCLIP loci as a TranscriptML RBPNet array bundle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from transcriptml.data.bundle import DatasetBundle, load_bundle, save_bundle_metadata
from transcriptml.data.encoding import encode_rna_sequence
from transcriptml.data.schemas import RNA4
from transcriptml.progress import ProgressReporter, log_progress
from transcriptml.rbpnet.experiment import ProcessedECLIPDataset
from transcriptml.rbpnet.selection import SelectionManifest, load_selection_manifest


@dataclass(frozen=True)
class RBPNetBundleConfig:
    """Configuration for fixed-shape RBPNet bundle materialization."""

    processed_dir: Path
    selection_manifest: Path
    output_dir: Path
    input_length: int = 300
    profile_length: int = 300
    max_jitter: int = 0
    transcript_end_policy: str = "drop"
    overwrite: bool = False
    progress: bool = True


def _materialized_interval(anchor: int, length: int, jitter: int) -> tuple[int, int]:
    width = length + 2 * jitter
    start = anchor - width // 2
    return start, start + width


def _source_and_destination(start: int, end: int, transcript_length: int) -> tuple[int, int, int, int]:
    source_start = max(0, start)
    source_end = min(transcript_length, end)
    destination_start = source_start - start
    destination_end = destination_start + max(0, source_end - source_start)
    return source_start, source_end, destination_start, destination_end


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    known = {
        "X.npy", "sminput_profiles.npy", "ip_profiles.npy",
        "sequence_valid_mask.npy", "profile_valid_mask.npy",
        "profile_sminput_totals.npy", "profile_ip_totals.npy",
        "selection_sminput_counts.npy", "selection_ip_counts.npy",
        "ids.txt", "metadata.json", "schema.json", "config.json", "examples.parquet",
    }
    existing = [path / name for name in known if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"bundle output already exists ({existing[0]}); pass --overwrite")
    for item in existing:
        item.unlink()


def _validate_config(config: RBPNetBundleConfig) -> None:
    if config.input_length <= 0 or config.profile_length <= 0:
        raise ValueError("input_length and profile_length must be positive")
    if config.max_jitter < 0:
        raise ValueError("max_jitter must be non-negative")
    if config.transcript_end_policy not in {"drop", "pad"}:
        raise ValueError("transcript_end_policy must be drop or pad")


def _sorted_rows(manifest: SelectionManifest) -> list[dict]:
    order = pc.sort_indices(manifest.table, sort_keys=[("example_id", "ascending")])
    return pc.take(manifest.table, order).to_pylist()


def _validate_manifest_dataset(
    manifest: SelectionManifest,
    ds: ProcessedECLIPDataset,
) -> None:
    scan = manifest.metadata.get("window_scan", {})
    observed = {
        str(name): int(value)
        for name, value in scan.get("effective_library_sizes", {}).items()
    }
    expected = {
        sample.name: int(sample.effective_library_size)
        for sample in ds.samples
    }
    if observed != expected:
        raise ValueError(
            "selection manifest effective library sizes do not match the processed experiment"
        )
    required = {
        f"{sample.name}_{suffix}"
        for sample in ds.samples
        for suffix in ("count", "cpm")
    }
    missing = sorted(required - set(manifest.table.column_names))
    if missing:
        raise ValueError(
            f"selection manifest lacks sample columns required by this experiment: {', '.join(missing)}"
        )


def make_rbpnet_bundle(config: RBPNetBundleConfig) -> DatasetBundle:
    """Materialize selected loci into memory-mappable NumPy arrays.

    ``input_length`` and ``profile_length`` describe future training crops.
    The stored widths add ``2 * max_jitter`` so a future loader can choose a
    shared positional shift without reopening FASTA or HDF5 files.
    """

    _validate_config(config)
    manifest = load_selection_manifest(config.selection_manifest)
    rows = _sorted_rows(manifest)
    sequence_width = config.input_length + 2 * config.max_jitter
    profile_width = config.profile_length + 2 * config.max_jitter
    _prepare_output(config.output_dir, config.overwrite)

    log_progress(
        f"rbpnet make-bundle: validate {len(rows):,} selected examples",
        enabled=config.progress,
    )
    with ProcessedECLIPDataset(config.processed_dir) as ds:
        _validate_manifest_dataset(manifest, ds)
        ip_samples = ds.ip_samples
        if not ip_samples:
            raise ValueError("processed dataset contains no IP samples")
        kept: list[dict] = []
        dropped = 0
        for row in rows:
            tx = ds.get_transcript(row["transcript_id"])
            anchor = int(row["transcript_anchor"])
            selection_start = int(row["selection_start"])
            selection_end = int(row["selection_end"])
            if selection_start < 0 or selection_end <= selection_start or selection_end > tx.length:
                raise ValueError(
                    f"invalid selection interval {tx.transcript_id}:{selection_start}-{selection_end}; "
                    f"transcript length is {tx.length}"
                )
            if anchor < 0 or anchor >= tx.length:
                raise ValueError(
                    f"selection anchor {anchor} is outside transcript {tx.transcript_id} length {tx.length}"
                )
            replicate_id = str(row["replicate_id"])
            if replicate_id and replicate_id not in {sample.name for sample in ip_samples}:
                raise ValueError(f"selection manifest has unknown IP replicate_id {replicate_id!r}")
            seq_start, seq_end = _materialized_interval(anchor, config.input_length, config.max_jitter)
            profile_start, profile_end = _materialized_interval(anchor, config.profile_length, config.max_jitter)
            in_bounds = (
                seq_start >= 0 and seq_end <= tx.length
                and profile_start >= 0 and profile_end <= tx.length
            )
            if config.transcript_end_policy == "drop" and not in_bounds:
                dropped += 1
                continue
            row = dict(row)
            row.update({
                "sequence_context_start": seq_start,
                "sequence_context_end": seq_end,
                "profile_context_start": profile_start,
                "profile_context_end": profile_end,
            })
            kept.append(row)
        if not kept:
            raise ValueError(
                "no examples remain after transcript-end handling; use --transcript-end-policy pad "
                "or reduce context/jitter lengths"
            )

        n_examples = len(kept)
        n_ip = len(ip_samples)
        X = np.lib.format.open_memmap(
            config.output_dir / "X.npy", mode="w+", dtype=np.uint8,
            shape=(n_examples, 4, sequence_width),
        )
        sminput_profiles = np.lib.format.open_memmap(
            config.output_dir / "sminput_profiles.npy", mode="w+", dtype=np.uint32,
            shape=(n_examples, profile_width),
        )
        ip_profiles = np.lib.format.open_memmap(
            config.output_dir / "ip_profiles.npy", mode="w+", dtype=np.uint32,
            shape=(n_examples, n_ip, profile_width),
        )
        sequence_valid_mask = np.lib.format.open_memmap(
            config.output_dir / "sequence_valid_mask.npy", mode="w+", dtype=np.uint8,
            shape=(n_examples, sequence_width),
        )
        profile_valid_mask = np.lib.format.open_memmap(
            config.output_dir / "profile_valid_mask.npy", mode="w+", dtype=np.uint8,
            shape=(n_examples, profile_width),
        )
        profile_sminput_totals = np.lib.format.open_memmap(
            config.output_dir / "profile_sminput_totals.npy", mode="w+", dtype=np.uint64,
            shape=(n_examples,),
        )
        profile_ip_totals = np.lib.format.open_memmap(
            config.output_dir / "profile_ip_totals.npy", mode="w+", dtype=np.uint64,
            shape=(n_examples, n_ip),
        )
        selection_sminput_counts = np.lib.format.open_memmap(
            config.output_dir / "selection_sminput_counts.npy", mode="w+", dtype=np.uint64,
            shape=(n_examples,),
        )
        selection_ip_counts = np.lib.format.open_memmap(
            config.output_dir / "selection_ip_counts.npy", mode="w+", dtype=np.uint64,
            shape=(n_examples, n_ip),
        )
        arrays = {
            "sminput_profiles": sminput_profiles,
            "ip_profiles": ip_profiles,
            "sequence_valid_mask": sequence_valid_mask,
            "profile_valid_mask": profile_valid_mask,
            "profile_sminput_totals": profile_sminput_totals,
            "profile_ip_totals": profile_ip_totals,
            "selection_sminput_counts": selection_sminput_counts,
            "selection_ip_counts": selection_ip_counts,
        }
        metadata: list[dict] = []
        reporter = ProgressReporter(
            "rbpnet make-bundle: materialize examples",
            total=n_examples,
            unit="examples",
            enabled=config.progress,
        )
        input_name = ds.sminput_sample.name
        for index, row in enumerate(kept):
            tx = ds.get_transcript(row["transcript_id"])
            seq_start = int(row["sequence_context_start"])
            seq_end = int(row["sequence_context_end"])
            src_start, src_end, dst_start, dst_end = _source_and_destination(
                seq_start, seq_end, tx.length
            )
            if src_end > src_start:
                encoded = encode_rna_sequence(ds.get_sequence(tx.transcript_id, src_start, src_end))
                X[index, :, dst_start:dst_end] = encoded
                sequence_valid_mask[index, dst_start:dst_end] = 1

            profile_start = int(row["profile_context_start"])
            profile_end = int(row["profile_context_end"])
            psrc_start, psrc_end, pdst_start, pdst_end = _source_and_destination(
                profile_start, profile_end, tx.length
            )
            if psrc_end > psrc_start:
                sminput = ds.get_profile(tx.transcript_id, psrc_start, psrc_end, input_name)
                sminput_profiles[index, pdst_start:pdst_end] = sminput
                for ip_index, sample in enumerate(ip_samples):
                    ip_profiles[index, ip_index, pdst_start:pdst_end] = ds.get_profile(
                        tx.transcript_id, psrc_start, psrc_end, sample.name
                    )
                profile_valid_mask[index, pdst_start:pdst_end] = 1
            profile_sminput_totals[index] = sminput_profiles[index].sum(dtype=np.uint64)
            profile_ip_totals[index] = ip_profiles[index].sum(axis=1, dtype=np.uint64)
            selection_sminput_counts[index] = int(row[f"{input_name}_count"])
            selection_ip_counts[index] = np.asarray(
                [int(row[f"{sample.name}_count"]) for sample in ip_samples], dtype=np.uint64
            )
            metadata.append({
                "example_id": row["example_id"],
                "gene_id": row["gene_id"],
                "transcript_id": row["transcript_id"],
                "chromosome": row["chromosome"],
                "strand": row["strand"],
                "transcript_anchor": int(row["transcript_anchor"]),
                "selection_start": int(row["selection_start"]),
                "selection_end": int(row["selection_end"]),
                "region_type": row["region_type"],
                "selection_strategy": row["selection_strategy"],
                "selection_state": row["selection_state"],
                "replicate_id": row["replicate_id"],
                "group_gene_id": row["group_gene_id"],
                "group_transcript_id": row["group_transcript_id"],
                "group_chromosome": row["group_chromosome"],
                "sequence_context_start": seq_start,
                "sequence_context_end": seq_end,
                "sequence_left_pad": max(0, -seq_start),
                "sequence_right_pad": max(0, seq_end - tx.length),
                "profile_context_start": profile_start,
                "profile_context_end": profile_end,
                "profile_left_pad": max(0, -profile_start),
                "profile_right_pad": max(0, profile_end - tx.length),
            })
            reporter.update()
        reporter.close()
        X.flush()
        for array in arrays.values():
            array.flush()

        # Keep a self-contained scalable copy of the selected-example contract,
        # sorted in the exact same stable-ID order as the arrays.
        example_table = pa.Table.from_pylist(kept, schema=manifest.table.schema)
        pq.write_table(example_table, config.output_dir / "examples.parquet", compression="zstd")
        config_payload = {
            "builder": "rbpnet",
            "bundle_format": "transcriptml-rbpnet-bundle",
            "bundle_format_version": "1",
            "source_processed_dir": str(config.processed_dir.resolve()),
            "source_selection_manifest": str(manifest.path.resolve()),
            "source_selection_sha256": _sha256(manifest.path),
            "selection": manifest.metadata,
            "input_length": config.input_length,
            "profile_length": config.profile_length,
            "max_jitter": config.max_jitter,
            "materialized_sequence_length": sequence_width,
            "materialized_profile_length": profile_width,
            "jitter_contract": (
                "future shift s in [-max_jitter,+max_jitter] takes sequence/profile crop "
                "starting at max_jitter+s from their respective materialized arrays"
            ),
            "transcript_end_policy": config.transcript_end_policy,
            "n_selected_manifest_rows": len(rows),
            "n_dropped_at_transcript_ends": dropped,
            "sample_metadata": {
                "sminput": {
                    "name": ds.sminput_sample.name,
                    "effective_library_size": ds.sminput_sample.effective_library_size,
                },
                "ip": [
                    {"name": sample.name, "effective_library_size": sample.effective_library_size}
                    for sample in ip_samples
                ],
                "ip_axis_order": [sample.name for sample in ip_samples],
                "pooled_ip_definition": "sum ip_profiles across axis 1",
            },
            "example_metadata_file": "examples.parquet",
        }
        bundle = DatasetBundle(
            X=X,
            y=None,
            ids=[row["example_id"] for row in kept],
            schema=RNA4,
            metadata=metadata,
            config=config_payload,
            arrays=arrays,
        )
        save_bundle_metadata(bundle, config.output_dir)
        log_progress(
            f"rbpnet make-bundle: wrote {n_examples:,} examples to {config.output_dir}",
            enabled=config.progress,
        )
        return bundle


def load_rbpnet_bundle(path: str | Path, *, mmap_mode: str | None = "r") -> DatasetBundle:
    """Load and validate a materialized RBPNet bundle."""

    bundle = load_bundle(path, mmap_mode=mmap_mode)
    if bundle.config.get("bundle_format") != "transcriptml-rbpnet-bundle":
        raise ValueError(f"not a TranscriptML RBPNet bundle: {path}")
    if str(bundle.config.get("bundle_format_version")) != "1":
        raise ValueError("unsupported RBPNet bundle format version")
    required = {
        "sminput_profiles", "ip_profiles", "sequence_valid_mask", "profile_valid_mask",
        "profile_sminput_totals", "profile_ip_totals",
        "selection_sminput_counts", "selection_ip_counts",
    }
    missing = sorted(required - set(bundle.arrays))
    if missing:
        raise ValueError(f"RBPNet bundle lacks named arrays: {', '.join(missing)}")
    return bundle
