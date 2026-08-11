"""Lazy, ergonomic access to a processed transcript-space eCLIP experiment."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pysam


@dataclass(frozen=True)
class RegionRecord:
    start: int
    end: int
    region_type: str


@dataclass(frozen=True)
class TranscriptRecord:
    gene_id: str
    transcript_id: str
    chromosome: str
    strand: str
    length: int
    signal_offset: int
    sminput_tpm: float
    regions: tuple[RegionRecord, ...]
    coordinate_space: str = "mature_transcript"
    genomic_start: int = 0
    genomic_end: int = 0


@dataclass(frozen=True)
class SampleRecord:
    name: str
    role: str
    effective_library_size: int | None


@dataclass(frozen=True)
class GenomicBlock:
    chromosome: str
    start: int
    end: int


class ProcessedECLIPDataset:
    """Thin lazy reader for one canonical processed eCLIP directory.

    Small metadata tables are loaded at construction. FASTA and HDF5 handles
    are opened only on first access and are never inherited when the reader is
    pickled, making the object safe to construct before worker processes.
    """

    SUPPORTED_FORMATS = {"transcriptml-rbpnet-experiment", "rbpnet-preprocess-dataset"}

    def __init__(self, processed_dir: str | Path):
        self.processed_dir = Path(processed_dir)
        manifest_path = self.processed_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_format = self.manifest.get("format")
        if data_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported processed experiment format {data_format!r}; "
                f"supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        if str(self.manifest.get("format_version")) != "1":
            raise ValueError(
                f"unsupported processed experiment format_version {self.manifest.get('format_version')!r}"
            )
        self.coordinate_space = self.manifest.get("coordinate_space", "mature_transcript")
        if self.coordinate_space not in {"mature_transcript", "gene"}:
            raise ValueError(f"unsupported coordinate_space {self.coordinate_space!r}")
        files = self.manifest.get("files", {})
        required = {"metadata", "exon_mapping", "sequences", "signals"}
        missing_keys = sorted(required - set(files))
        if missing_keys:
            raise ValueError(f"manifest is missing file entries: {', '.join(missing_keys)}")
        self._metadata_path = self.processed_dir / files["metadata"]
        self._exon_path = self.processed_dir / files["exon_mapping"]
        self._fasta_path = self.processed_dir / files["sequences"]
        self._signal_path = self.processed_dir / files["signals"]
        for path in (self._metadata_path, self._exon_path, self._fasta_path, self._signal_path):
            if not path.is_file():
                raise FileNotFoundError(f"processed dataset file not found: {path}")

        self.transcripts = self._read_transcripts()
        self._transcript_by_id = {tx.transcript_id: tx for tx in self.transcripts}
        if len(self._transcript_by_id) != len(self.transcripts):
            raise ValueError("duplicate transcript IDs in metadata")
        self.samples = tuple(
            SampleRecord(
                name=sample["name"],
                role=sample["role"],
                effective_library_size=(
                    int(sample["effective_library_size"])
                    if "effective_library_size" in sample else None
                ),
            )
            for sample in self.manifest.get("samples", [])
        )
        self._sample_index = {sample.name: index for index, sample in enumerate(self.samples)}
        if len(self._sample_index) != len(self.samples):
            raise ValueError("duplicate sample names in manifest")
        self._fasta: pysam.FastaFile | None = None
        self._h5: h5py.File | None = None
        self._exons_by_transcript: dict[str, list[dict]] | None = None
        self._validate_store()

    def _read_transcripts(self) -> tuple[TranscriptRecord, ...]:
        records = []
        with self._metadata_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                regions = tuple(
                    RegionRecord(int(region["start"]), int(region["end"]), region["type"])
                    for region in json.loads(row["region_annotations"])
                )
                record = TranscriptRecord(
                    gene_id=row["gene_id"],
                    transcript_id=row["transcript_id"],
                    chromosome=row["chrom"],
                    strand=row["strand"],
                    coordinate_space=row.get("coordinate_space", self.coordinate_space),
                    genomic_start=int(row.get("genomic_start") or 0),
                    genomic_end=int(row.get("genomic_end") or 0),
                    length=int(row["transcript_length"]),
                    signal_offset=int(row["signal_offset"]),
                    sminput_tpm=float(row["sm_input_tpm"]),
                    regions=regions,
                )
                if record.length <= 0:
                    raise ValueError(f"transcript {record.transcript_id} has non-positive length")
                if record.coordinate_space != self.coordinate_space:
                    raise ValueError(
                        f"metadata coordinate space differs for {record.transcript_id}"
                    )
                if record.coordinate_space == "gene" and (
                    record.genomic_start < 0
                    or record.genomic_end - record.genomic_start != record.length
                ):
                    raise ValueError(
                        f"invalid gene span for {record.transcript_id}: "
                        f"{record.genomic_start}-{record.genomic_end}"
                    )
                if (
                    not regions
                    or regions[0].start != 0
                    or regions[-1].end != record.length
                    or any(
                        region.start < 0
                        or region.end <= region.start
                        or (index and regions[index - 1].end != region.start)
                        for index, region in enumerate(regions)
                    )
                ):
                    raise ValueError(
                        f"region annotations do not partition transcript {record.transcript_id}"
                    )
                records.append(record)
        return tuple(records)

    @staticmethod
    def _decode(values) -> list[str]:
        return [value.decode() if isinstance(value, bytes) else str(value) for value in values]

    def _open_handles(self) -> None:
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self._fasta_path))
        if self._h5 is None:
            self._h5 = h5py.File(self._signal_path, "r")

    def _validate_store(self) -> None:
        with h5py.File(self._signal_path, "r") as store:
            required = {
                "counts", "ip_pooled", "sample_names", "sample_roles", "transcript_ids",
                "transcript_offsets", "transcript_lengths",
            }
            missing = sorted(required - set(store))
            if missing:
                raise ValueError(f"signals.h5 is missing datasets: {', '.join(missing)}")
            h5_samples = self._decode(store["sample_names"][:])
            if h5_samples != [sample.name for sample in self.samples]:
                raise ValueError("sample order differs between manifest and signals.h5")
            h5_roles = self._decode(store["sample_roles"][:])
            if h5_roles != [sample.role for sample in self.samples]:
                raise ValueError("sample roles differ between manifest and signals.h5")
            h5_transcripts = self._decode(store["transcript_ids"][:])
            if h5_transcripts != [tx.transcript_id for tx in self.transcripts]:
                raise ValueError("transcript order differs between metadata and signals.h5")
            expected_offsets = np.asarray([tx.signal_offset for tx in self.transcripts], dtype=np.int64)
            expected_lengths = np.asarray([tx.length for tx in self.transcripts], dtype=np.int64)
            if not np.array_equal(store["transcript_offsets"][:], expected_offsets):
                raise ValueError("transcript offsets differ between metadata and signals.h5")
            if not np.array_equal(store["transcript_lengths"][:], expected_lengths):
                raise ValueError("transcript lengths differ between metadata and signals.h5")
            total_length = int(expected_lengths.sum())
            if store["counts"].shape != (len(self.samples), total_length):
                raise ValueError("counts shape disagrees with sample and transcript metadata")
            if store["ip_pooled"].shape != (total_length,):
                raise ValueError("ip_pooled shape disagrees with transcript metadata")
            stored_space = store.attrs.get("coordinate_space")
            if isinstance(stored_space, bytes):
                stored_space = stored_space.decode()
            if stored_space is not None and str(stored_space) != self.coordinate_space:
                raise ValueError("coordinate space differs between manifest and signals.h5")
        with pysam.FastaFile(str(self._fasta_path)) as fasta:
            if tuple(fasta.references) != tuple(tx.transcript_id for tx in self.transcripts):
                raise ValueError("transcript order differs between metadata and transcript FASTA")
            if tuple(fasta.lengths) != tuple(tx.length for tx in self.transcripts):
                raise ValueError("transcript lengths differ between metadata and transcript FASTA")

    @property
    def sample_names(self) -> tuple[str, ...]:
        return tuple(sample.name for sample in self.samples)

    @property
    def ip_samples(self) -> tuple[SampleRecord, ...]:
        return tuple(sample for sample in self.samples if sample.role == "ip")

    @property
    def sminput_sample(self) -> SampleRecord:
        inputs = [sample for sample in self.samples if sample.role == "sminput"]
        if len(inputs) != 1:
            raise ValueError(f"expected exactly one sminput sample, found {len(inputs)}")
        return inputs[0]

    @property
    def pooled_ip_effective_library_size(self) -> int:
        sizes = [sample.effective_library_size for sample in self.ip_samples]
        if not sizes or any(size is None for size in sizes):
            raise ValueError("manifest lacks effective library size for one or more IP samples")
        return sum(int(size) for size in sizes)

    def get_transcript(self, transcript_id: str) -> TranscriptRecord:
        try:
            return self._transcript_by_id[transcript_id]
        except KeyError as exc:
            raise KeyError(f"unknown transcript: {transcript_id}") from exc

    def _slice(self, transcript_id: str, start: int, end: int) -> tuple[TranscriptRecord, slice]:
        tx = self.get_transcript(transcript_id)
        if start < 0 or end < start or end > tx.length:
            raise IndexError(
                f"invalid interval {transcript_id}:{start}-{end}; transcript length is {tx.length}"
            )
        return tx, slice(tx.signal_offset + start, tx.signal_offset + end)

    def get_sequence(self, transcript_id: str, start: int, end: int) -> str:
        self._slice(transcript_id, start, end)
        self._open_handles()
        assert self._fasta is not None
        return self._fasta.fetch(transcript_id, start, end)

    def get_profile(self, transcript_id: str, start: int, end: int, sample: str) -> np.ndarray:
        _, flat_slice = self._slice(transcript_id, start, end)
        try:
            sample_index = self._sample_index[sample]
        except KeyError as exc:
            raise KeyError(f"unknown sample {sample!r}; available: {', '.join(self.sample_names)}") from exc
        self._open_handles()
        assert self._h5 is not None
        return self._h5["counts"][sample_index, flat_slice]

    def get_profiles(self, transcript_id: str, start: int, end: int) -> np.ndarray:
        """Return all sample profiles in manifest order for one interval."""

        _, flat_slice = self._slice(transcript_id, start, end)
        self._open_handles()
        assert self._h5 is not None
        return self._h5["counts"][:, flat_slice]

    def get_pooled_ip_profile(self, transcript_id: str, start: int, end: int) -> np.ndarray:
        _, flat_slice = self._slice(transcript_id, start, end)
        self._open_handles()
        assert self._h5 is not None
        return self._h5["ip_pooled"][flat_slice]

    def _load_exons(self) -> None:
        exons: dict[str, list[dict]] = {}
        with gzip.open(self._exon_path, "rt") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                exons.setdefault(row["transcript_id"], []).append({
                    "tx_start": int(row["tx_start"]),
                    "tx_end": int(row["tx_end"]),
                    "chromosome": row["chrom"],
                    "genomic_start": int(row["genomic_start"]),
                    "genomic_end": int(row["genomic_end"]),
                    "strand": row["strand"],
                })
        self._exons_by_transcript = exons

    def get_genomic_blocks(
        self, transcript_id: str, start: int, end: int
    ) -> tuple[GenomicBlock, ...]:
        """Map one locus interval to compact ascending genomic blocks."""

        tx, _ = self._slice(transcript_id, start, end)
        if self.coordinate_space == "gene":
            if tx.strand == "+":
                genomic_start = tx.genomic_start + start
                genomic_end = tx.genomic_start + end
            else:
                genomic_start = tx.genomic_end - end
                genomic_end = tx.genomic_end - start
            return (GenomicBlock(tx.chromosome, genomic_start, genomic_end),)
        if self._exons_by_transcript is None:
            self._load_exons()
        assert self._exons_by_transcript is not None
        blocks = []
        for exon in self._exons_by_transcript.get(transcript_id, []):
            lo = max(start, exon["tx_start"])
            hi = min(end, exon["tx_end"])
            if lo >= hi:
                continue
            if tx.strand == "+":
                genomic_start = exon["genomic_start"] + lo - exon["tx_start"]
                genomic_end = exon["genomic_start"] + hi - exon["tx_start"]
            else:
                genomic_start = exon["genomic_end"] - (hi - exon["tx_start"])
                genomic_end = exon["genomic_end"] - (lo - exon["tx_start"])
            blocks.append(GenomicBlock(exon["chromosome"], genomic_start, genomic_end))
        if sum(block.end - block.start for block in blocks) != end - start:
            raise ValueError(f"exon mapping does not cover {transcript_id}:{start}-{end}")
        return tuple(blocks)

    def coordinate_to_genome(self, transcript_id: str, pos: int) -> tuple[str, int, str]:
        """Map one selected-coordinate-space base to a genomic base."""

        tx, _ = self._slice(transcript_id, pos, pos + 1)
        if self.coordinate_space == "gene":
            genomic = (
                tx.genomic_start + pos
                if tx.strand == "+"
                else tx.genomic_end - 1 - pos
            )
            return tx.chromosome, genomic, tx.strand
        if self._exons_by_transcript is None:
            self._load_exons()
        assert self._exons_by_transcript is not None
        for exon in self._exons_by_transcript.get(transcript_id, []):
            if exon["tx_start"] <= pos < exon["tx_end"]:
                offset = pos - exon["tx_start"]
                genomic = (
                    exon["genomic_start"] + offset
                    if tx.strand == "+"
                    else exon["genomic_end"] - 1 - offset
                )
                return tx.chromosome, genomic, tx.strand
        raise ValueError(f"exon mapping does not cover {transcript_id}:{pos}")

    def genome_to_coordinate(self, transcript_id: str, chromosome: str, pos: int) -> int | None:
        """Map one genomic base into the selected coordinate space, if represented."""

        tx = self.get_transcript(transcript_id)
        if chromosome != tx.chromosome:
            return None
        if self.coordinate_space == "gene":
            if not tx.genomic_start <= pos < tx.genomic_end:
                return None
            return (
                pos - tx.genomic_start
                if tx.strand == "+"
                else tx.genomic_end - 1 - pos
            )
        if self._exons_by_transcript is None:
            self._load_exons()
        assert self._exons_by_transcript is not None
        for exon in self._exons_by_transcript.get(transcript_id, []):
            if exon["genomic_start"] <= pos < exon["genomic_end"]:
                offset = (
                    pos - exon["genomic_start"]
                    if tx.strand == "+"
                    else exon["genomic_end"] - 1 - pos
                )
                return exon["tx_start"] + offset
        return None

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_fasta"] = None
        return state

    def __enter__(self) -> "ProcessedECLIPDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
