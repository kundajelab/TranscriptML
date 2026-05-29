from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np
import torch

from transcriptml.data.encoding import infer_valid_lengths
from transcriptml.data.schemas import SequenceSchema, get_schema
from transcriptml.interpret.predictor import Predictor
from transcriptml.progress import ProgressReporter, log_progress

MutationPolicy = Literal["synonymous", "synonymous-only", "all", "all-codons"]

AA_TO_CODONS: dict[str, tuple[str, ...]] = {
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
CODON_TO_AA = {codon: aa for aa, codons in AA_TO_CODONS.items() for codon in codons}
ALL_CODONS = tuple(codon for codons in AA_TO_CODONS.values() for codon in codons)
STOP_CODONS = AA_TO_CODONS["Stop"]
SENSE_CODONS = tuple(codon for codon in ALL_CODONS if codon not in STOP_CODONS)


@dataclass(frozen=True)
class CDSCodonStarts:
    """CDS coordinates and codon-start positions for one encoded transcript."""

    starts: np.ndarray
    cds_start: int
    cds_end: int
    cds_length: int
    encoding: str


@dataclass(frozen=True)
class CodonMutation:
    """One row in the long-form codon mutagenesis table."""

    sequence_index: int
    codon_start: int
    cds_codon_index: int
    cds_start: int
    cds_end: int
    cds_length: int
    cds_relative_position: float
    reference_codon: str
    alternate_codon: str
    reference_amino_acid: str
    alternate_amino_acid: str
    synonymous: bool
    reference_prediction: float
    mutant_prediction: float
    delta: float


@dataclass
class CodonISMResult:
    """Result from codon-level mutational scanning."""

    mutations: np.ndarray
    reference_predictions: np.ndarray
    valid_lengths: np.ndarray
    sequence_indices: np.ndarray
    position_scores: np.ndarray | None = None


class MutationTableWriter(Protocol):
    """Streaming sink for long-form codon mutation rows."""

    def write(self, rows: Sequence[CodonMutation]) -> None:
        """Write a chunk of mutation rows.

        Args:
            rows: Sequence of long-form codon mutation records to append.
        """

    def close(self) -> None:
        """Flush and close the writer."""


_MUTATION_DTYPE = np.dtype(
    [
        ("sequence_index", np.int64),
        ("codon_start", np.int64),
        ("cds_codon_index", np.int64),
        ("cds_start", np.int64),
        ("cds_end", np.int64),
        ("cds_length", np.int64),
        ("cds_relative_position", np.float32),
        ("reference_codon", "U3"),
        ("alternate_codon", "U3"),
        ("reference_amino_acid", "U4"),
        ("alternate_amino_acid", "U4"),
        ("synonymous", np.bool_),
        ("reference_prediction", np.float32),
        ("mutant_prediction", np.float32),
        ("delta", np.float32),
    ]
)
_ROW_FIELDS = tuple(field.name for field in fields(CodonMutation))


def _normalise_policy(policy: str) -> Literal["synonymous", "all"]:
    """Normalize mutation-policy aliases to internal policy names.

    Args:
        policy: User-facing mutation policy string.
    """

    key = str(policy).lower().replace("_", "-")
    if key in {"synonymous", "synonymous-only", "synonymousonly"}:
        return "synonymous"
    if key in {"all", "all-codons", "allcodons"}:
        return "all"
    raise ValueError("mutation_policy must be 'synonymous-only'/'synonymous' or 'all-codons'/'all'")


def codon_alternates(
    reference_codon: str,
    *,
    mutation_policy: MutationPolicy = "synonymous-only",
    include_stop_codons: bool = True,
) -> tuple[str, ...]:
    """Return alternate codons for a reference codon under a mutation policy.

    Args:
        reference_codon: Three-base reference codon using RNA or DNA alphabet.
        mutation_policy: Alternate-codon policy, either synonymous-only or all
            codons.
        include_stop_codons: Whether all-codon mode may include stop codons.
    """

    ref = reference_codon.upper().replace("T", "U")
    policy = _normalise_policy(mutation_policy)
    if ref not in CODON_TO_AA:
        return ()
    if policy == "synonymous":
        aa = CODON_TO_AA[ref]
        return tuple(codon for codon in AA_TO_CODONS[aa] if codon != ref)
    pool = ALL_CODONS if include_stop_codons else SENSE_CODONS
    return tuple(codon for codon in pool if codon != ref)


def _as_numpy_tensor(X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert supported tensor inputs to a NumPy array.

    Args:
        X: NumPy array or torch tensor to convert.
    """

    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _predict(
    predictor: Predictor | Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Run prediction through a Predictor-like object or callable.

    Args:
        predictor: Predictor-like object with ``predict`` or callable accepting
            a NumPy batch.
        X: Encoded batch array to score.
        batch_size: Optional prediction batch-size hint for Predictor-like
            objects.
    """

    if hasattr(predictor, "predict"):
        try:
            pred = predictor.predict(X, batch_size=batch_size)  # type: ignore[attr-defined]
        except TypeError:
            pred = predictor.predict(X)  # type: ignore[attr-defined]
    else:
        pred = predictor(X)
    return np.asarray(pred, dtype=np.float32).reshape(-1)


def _resolve_cds_channel(schema: SequenceSchema, cds_channel: str | int | None) -> int:
    """Resolve a CDS channel selector to an integer channel index.

    Args:
        schema: Sequence schema describing available channels.
        cds_channel: Optional explicit CDS channel index or name. When omitted,
            a CDS-like channel is inferred from ``schema``.
    """

    if isinstance(cds_channel, int):
        if cds_channel < 0 or cds_channel >= schema.n_channels:
            raise ValueError(f"cds_channel index {cds_channel} is outside schema with {schema.n_channels} channels")
        return int(cds_channel)
    if isinstance(cds_channel, str):
        try:
            return schema.channels.index(cds_channel)
        except ValueError as exc:
            raise ValueError(f"cds_channel '{cds_channel}' is not in schema channels {schema.channels}") from exc

    preferred = ("CDS_codon_start", "cds_codon_start", "codon_start", "CDS", "cds")
    lower_to_index = {name.lower(): i for i, name in enumerate(schema.channels)}
    for name in preferred:
        if name.lower() in lower_to_index:
            return lower_to_index[name.lower()]
    for i, name in enumerate(schema.channels):
        lowered = name.lower()
        if "cds" in lowered or "coding" in lowered or "codon_start" in lowered:
            return i
    raise ValueError("Could not infer CDS channel from schema; pass cds_channel explicitly")


def _base_channels(schema: SequenceSchema) -> tuple[np.ndarray, tuple[str, ...], dict[str, int], dict[str, int]]:
    """Resolve base-channel metadata from a sequence schema.

    Args:
        schema: Sequence schema expected to contain exactly one A, C, G, and U/T
            base channel.
    """

    channel_indices: list[int] = []
    letters: list[str] = []
    for base_name in schema.base_channels:
        if base_name not in schema.channels:
            raise ValueError(f"Base channel '{base_name}' is not present in schema channels {schema.channels}")
        letter = base_name.upper().replace("T", "U")
        if letter not in {"A", "C", "G", "U"}:
            raise ValueError(f"Unsupported base channel '{base_name}'; expected A/C/G/U/T")
        channel_indices.append(schema.channels.index(base_name))
        letters.append(letter)
    if len(set(letters)) != 4 or set(letters) != {"A", "C", "G", "U"}:
        raise ValueError("Codon ISM requires exactly one A, C, G, and U/T base channel")
    base_to_channel = {base: int(idx) for base, idx in zip(letters, channel_indices)}
    base_to_offset = {base: offset for offset, base in enumerate(letters)}
    return np.asarray(channel_indices, dtype=np.int64), tuple(letters), base_to_channel, base_to_offset


def find_cds_codon_starts(
    x: np.ndarray | torch.Tensor,
    schema: str | SequenceSchema = "saluki6",
    *,
    valid_length: int | None = None,
    cds_channel: str | int | None = None,
) -> CDSCodonStarts:
    """Identify coding codon starts from a CDS annotation channel.

    Both sparse codon-start channels and dense all-CDS-position channels are
    supported. The dense/sparse choice is inferred from annotation spacing,
    matching the behavior of the legacy script while keeping the schema lookup
    explicit.

    Args:
        x: Encoded ``(C, L)`` transcript as a NumPy array or torch tensor.
        schema: Sequence schema name or object describing channel layout.
        valid_length: Optional valid transcript length to inspect. When omitted,
            the full encoded length is used.
        cds_channel: Optional CDS channel name or integer index. When omitted,
            the CDS channel is inferred from ``schema``.
    """

    seq = _as_numpy_tensor(x)
    if seq.ndim != 2:
        raise ValueError(f"Expected one transcript with shape (C, L), got {seq.shape}")
    resolved = get_schema(schema)
    channel = _resolve_cds_channel(resolved, cds_channel)
    L = int(seq.shape[-1])
    seq_len = L if valid_length is None else min(max(int(valid_length), 0), L)
    if seq_len <= 0:
        return CDSCodonStarts(np.empty((0,), dtype=np.int64), -1, -1, 0, "none")

    locs = np.flatnonzero(np.asarray(seq[channel, :seq_len]) > 0)
    if locs.size == 0:
        return CDSCodonStarts(np.empty((0,), dtype=np.int64), -1, -1, 0, "none")

    cds_start = int(locs[0])
    if locs.size >= 2:
        median_diff = int(np.median(np.diff(locs)))
        dense = median_diff == 1
    else:
        dense = False

    if dense:
        cds_end = int(locs[-1])
        starts = np.arange(cds_start, cds_end + 1, 3, dtype=np.int64)
        encoding = "dense"
    else:
        cds_end = min(int(locs[-1]) + 2, seq_len - 1)
        starts = locs.astype(np.int64, copy=False)
        encoding = "codon_start"

    if cds_end < cds_start:
        return CDSCodonStarts(np.empty((0,), dtype=np.int64), -1, -1, 0, "none")
    starts = starts[(starts >= cds_start) & (starts + 2 <= cds_end)]
    cds_length = int(cds_end - cds_start + 1)
    return CDSCodonStarts(starts.astype(np.int64, copy=False), cds_start, cds_end, cds_length, encoding)


def _decode_codon(
    x: np.ndarray,
    start: int,
    *,
    base_channel_indices: np.ndarray,
    base_letters: Sequence[str],
) -> tuple[str, tuple[int, int, int]] | None:
    """Decode one unambiguous codon from base channels.

    Args:
        x: Encoded ``(C, L)`` sequence array.
        start: Zero-based codon start position.
        base_channel_indices: Channel indices corresponding to A/C/G/U bases.
        base_letters: Base letters aligned to ``base_channel_indices``.
    """

    letters: list[str] = []
    offsets: list[int] = []
    for pos in range(int(start), int(start) + 3):
        values = np.asarray(x[base_channel_indices, pos])
        hits = np.flatnonzero(values > 0)
        if hits.size != 1:
            return None
        offset = int(hits[0])
        letters.append(base_letters[offset])
        offsets.append(offset)
    codon = "".join(letters)
    if codon not in CODON_TO_AA:
        return None
    return codon, (offsets[0], offsets[1], offsets[2])


def _write_codon_inplace(
    x: np.ndarray,
    start: int,
    codon: str,
    *,
    base_channel_indices: np.ndarray,
    base_to_channel: dict[str, int],
) -> None:
    """Overwrite one codon in an encoded sequence in place.

    Args:
        x: Encoded ``(C, L)`` sequence array to modify.
        start: Zero-based codon start position.
        codon: Three-base RNA codon to write.
        base_channel_indices: Channel indices for all base channels to clear.
        base_to_channel: Mapping from base letter to channel index.
    """

    x[base_channel_indices, int(start) : int(start) + 3] = 0
    for j, base in enumerate(codon):
        x[base_to_channel[base], int(start) + j] = 1


@dataclass(frozen=True)
class _PendingMutation:
    sequence_index: int
    local_index: int
    codon_start: int
    cds_codon_index: int
    cds_start: int
    cds_end: int
    cds_length: int
    cds_relative_position: float
    reference_codon: str
    alternate_codon: str
    reference_amino_acid: str
    alternate_amino_acid: str
    synonymous: bool
    reference_prediction: float


def _rows_to_structured(rows: Sequence[CodonMutation]) -> np.ndarray:
    """Convert mutation rows to the structured NumPy mutation dtype.

    Args:
        rows: Codon mutation records to convert.
    """

    arr = np.empty(len(rows), dtype=_MUTATION_DTYPE)
    for i, row in enumerate(rows):
        arr[i] = tuple(getattr(row, name) for name in _ROW_FIELDS)
    return arr


def _rows_to_columns(rows: Sequence[CodonMutation]) -> dict[str, np.ndarray]:
    """Convert mutation rows to a column dictionary.

    Args:
        rows: Codon mutation records to convert.
    """

    structured = _rows_to_structured(rows)
    return {name: np.asarray(structured[name]) for name in structured.dtype.names or ()}


def _structured_to_columns(table: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a structured mutation table to a column dictionary.

    Args:
        table: Structured NumPy mutation table matching ``_MUTATION_DTYPE``.
    """

    arr = np.asarray(table, dtype=_MUTATION_DTYPE)
    return {name: np.asarray(arr[name]) for name in arr.dtype.names or ()}


class CsvMutationTableWriter:
    """Stream codon mutation rows to a CSV file."""

    def __init__(self, path: str | Path):
        """Create a CSV mutation-table writer.

        Args:
            path: Destination CSV path.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(_ROW_FIELDS))
        self._writer.writeheader()

    def write(self, rows: Sequence[CodonMutation]) -> None:
        """Append mutation rows to the CSV file.

        Args:
            rows: Codon mutation records to write.
        """

        for row in rows:
            self._writer.writerow(asdict(row))

    def close(self) -> None:
        self._handle.close()


class NpzMutationTableWriter:
    """Stream codon mutation rows to a directory of compressed NPZ shards."""

    def __init__(self, path: str | Path, *, rows_per_shard: int = 100_000, compressed: bool = True):
        """Create a sharded NPZ mutation-table writer.

        Args:
            path: Destination directory for NPZ shards and manifest.
            rows_per_shard: Maximum number of mutation rows per shard.
            compressed: Whether to write compressed NPZ shards.
        """

        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = int(rows_per_shard)
        self.compressed = bool(compressed)
        self._buffer: list[CodonMutation] = []
        self._part = 0
        self._parts: list[dict[str, Any]] = []

    def write(self, rows: Sequence[CodonMutation]) -> None:
        """Buffer and write mutation rows to NPZ shards.

        Args:
            rows: Codon mutation records to append.
        """

        self._buffer.extend(rows)
        while len(self._buffer) >= self.rows_per_shard:
            self._write_shard(self._buffer[: self.rows_per_shard])
            del self._buffer[: self.rows_per_shard]

    def _write_shard(self, rows: Sequence[CodonMutation]) -> None:
        """Write one NPZ shard from buffered mutation rows.

        Args:
            rows: Codon mutation records for a single shard.
        """

        filename = f"part-{self._part:06d}.npz"
        arrays = _rows_to_columns(rows)
        save = np.savez_compressed if self.compressed else np.savez
        save(self.path / filename, **arrays)
        self._parts.append({"path": filename, "n_rows": int(len(rows))})
        self._part += 1

    def close(self) -> None:
        if self._buffer:
            self._write_shard(self._buffer)
            self._buffer.clear()
        manifest = {
            "format": "chunked_npz",
            "columns": list(_ROW_FIELDS),
            "dtype": {name: str(dtype) for name, (dtype, _) in _MUTATION_DTYPE.fields.items()},
            "parts": self._parts,
            "n_rows": int(sum(part["n_rows"] for part in self._parts)),
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _arrow_schema(pa: Any) -> Any:
    """Build the Arrow schema for codon mutation rows.

    Args:
        pa: Imported ``pyarrow`` module.
    """

    return pa.schema(
        [
            ("sequence_index", pa.int64()),
            ("codon_start", pa.int64()),
            ("cds_codon_index", pa.int64()),
            ("cds_start", pa.int64()),
            ("cds_end", pa.int64()),
            ("cds_length", pa.int64()),
            ("cds_relative_position", pa.float32()),
            ("reference_codon", pa.string()),
            ("alternate_codon", pa.string()),
            ("reference_amino_acid", pa.string()),
            ("alternate_amino_acid", pa.string()),
            ("synonymous", pa.bool_()),
            ("reference_prediction", pa.float32()),
            ("mutant_prediction", pa.float32()),
            ("delta", pa.float32()),
        ]
    )


def _rows_to_arrow_table(rows: Sequence[CodonMutation], pa: Any) -> Any:
    """Convert mutation rows to a PyArrow table.

    Args:
        rows: Codon mutation records to convert.
        pa: Imported ``pyarrow`` module.
    """

    columns = _rows_to_columns(rows)
    data = {name: columns[name] for name in _ROW_FIELDS}
    return pa.table(data, schema=_arrow_schema(pa))


class ParquetMutationTableWriter:
    """Stream codon mutation rows to a Parquet file using pyarrow."""

    def __init__(self, path: str | Path, *, compression: str = "zstd"):
        """Create a Parquet mutation-table writer.

        Args:
            path: Destination Parquet path.
            compression: PyArrow compression codec name.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._writer: Any | None = None
        self._pa: Any | None = None
        self._pq: Any | None = None

    def _imports(self) -> tuple[Any, Any]:
        if self._pa is None or self._pq is None:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet codon ISM output requires the optional 'pyarrow' package") from exc
            self._pa = pa
            self._pq = pq
        return self._pa, self._pq

    def write(self, rows: Sequence[CodonMutation]) -> None:
        """Append mutation rows to the Parquet file.

        Args:
            rows: Codon mutation records to write.
        """

        if not rows:
            return
        pa, pq = self._imports()
        table = _rows_to_arrow_table(rows, pa)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression=self.compression)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return
        pa, pq = self._imports()
        pq.write_table(pa.table({name: [] for name in _ROW_FIELDS}, schema=_arrow_schema(pa)), self.path)


class ArrowMutationTableWriter:
    """Stream codon mutation rows to an Arrow IPC file using pyarrow."""

    def __init__(self, path: str | Path):
        """Create an Arrow IPC mutation-table writer.

        Args:
            path: Destination Arrow IPC path.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pa: Any | None = None
        self._sink: Any | None = None
        self._writer: Any | None = None

    def _import_pa(self) -> Any:
        if self._pa is None:
            try:
                import pyarrow as pa
            except ImportError as exc:
                raise ImportError("Arrow codon ISM output requires the optional 'pyarrow' package") from exc
            self._pa = pa
        return self._pa

    def write(self, rows: Sequence[CodonMutation]) -> None:
        """Append mutation rows to the Arrow IPC file.

        Args:
            rows: Codon mutation records to write.
        """

        if not rows:
            return
        pa = self._import_pa()
        table = _rows_to_arrow_table(rows, pa)
        if self._writer is None:
            self._sink = pa.OSFile(str(self.path), "wb")
            self._writer = pa.ipc.new_file(self._sink, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        pa = self._import_pa()
        if self._writer is None:
            self._sink = pa.OSFile(str(self.path), "wb")
            self._writer = pa.ipc.new_file(self._sink, _arrow_schema(pa))
        self._writer.close()
        self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None


def mutation_table_writer(
    path: str | Path,
    *,
    format: Literal["auto", "csv", "npz", "parquet", "arrow"] = "auto",
    rows_per_shard: int = 100_000,
) -> MutationTableWriter:
    """Create a streaming mutation-table writer from a path and format.

    Args:
        path: Output file path or NPZ shard directory, depending on format.
        format: Output format. ``auto`` infers from ``path`` suffix and falls
            back to chunked NPZ.
        rows_per_shard: Maximum rows per NPZ shard when ``format`` is ``npz``
            or resolves to NPZ.
    """

    out = Path(path)
    fmt = format
    if fmt == "auto":
        suffix = out.suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix == ".parquet":
            fmt = "parquet"
        elif suffix in {".arrow", ".ipc", ".feather"}:
            fmt = "arrow"
        else:
            fmt = "npz"
    if fmt == "csv":
        return CsvMutationTableWriter(out)
    if fmt == "npz":
        return NpzMutationTableWriter(out, rows_per_shard=rows_per_shard)
    if fmt == "parquet":
        return ParquetMutationTableWriter(out)
    if fmt == "arrow":
        return ArrowMutationTableWriter(out)
    raise ValueError(f"Unknown mutation table format '{format}'")


@torch.no_grad()
def compute_codon_ism(
    X: np.ndarray | torch.Tensor,
    predictor: Predictor | torch.nn.Module | Callable[[np.ndarray], np.ndarray],
    *,
    schema: str | SequenceSchema = "saluki6",
    valid_lengths: Sequence[int] | None = None,
    cds_channel: str | int | None = None,
    mutation_policy: MutationPolicy = "synonymous-only",
    include_stop_codons: bool = True,
    reference_batch_size: int | None = None,
    mutation_batch_size: int = 512,
    compute_position_scores: bool = False,
    writer: MutationTableWriter | None = None,
    collect: bool = True,
    sequence_indices: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> CodonISMResult:
    """Run codon-level ISM and return mutant-minus-reference effects.

    The primary output is a tidy long-form mutation table in
    ``result.mutations``. For large analyses, pass a ``writer`` and
    ``collect=False`` to stream chunks without retaining every mutation row in
    memory.

    Args:
        X: Encoded ``(N, C, L)`` sequence batch as a NumPy array or torch tensor.
        predictor: Predictor, PyTorch module, or callable used to score
            reference and mutant sequences.
        schema: Sequence schema name or object describing channel layout.
        valid_lengths: Optional valid lengths for each sequence. When omitted,
            lengths are inferred from ``X``.
        cds_channel: Optional CDS channel name or integer index. When omitted,
            the CDS channel is inferred from ``schema``.
        mutation_policy: Alternate-codon policy, either synonymous-only or all
            codons.
        include_stop_codons: Whether all-codon mode may include stop-codon
            alternates.
        reference_batch_size: Optional batch size for reference predictions.
        mutation_batch_size: Maximum number of mutant sequences to score in one
            prediction batch.
        compute_position_scores: Whether to compute per-position max-absolute
            codon effects.
        writer: Optional streaming writer for long-form mutation rows.
        collect: Whether to retain mutation rows in ``result.mutations``.
        sequence_indices: Optional subset of sequence indices to analyze.
        device: Torch device used when ``predictor`` is a raw PyTorch module.
        progress: Whether to emit progress messages while scanning.
    """

    arr = _as_numpy_tensor(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    resolved_schema = get_schema(schema)
    if arr.shape[1] < resolved_schema.n_channels:
        raise ValueError(
            f"X has {arr.shape[1]} channels, but schema '{resolved_schema.name}' expects "
            f"{resolved_schema.n_channels}"
        )
    if mutation_batch_size <= 0:
        raise ValueError("mutation_batch_size must be positive")

    if isinstance(predictor, torch.nn.Module):
        predictor = Predictor(predictor, device=device, batch_size=reference_batch_size or 128)

    N, _, L = arr.shape
    lengths = (
        infer_valid_lengths(arr)
        if valid_lengths is None
        else np.asarray(valid_lengths, dtype=np.int64)
    )
    if lengths.shape[0] != N:
        raise ValueError("valid_lengths length must match X.shape[0]")

    analysis_indices = (
        np.arange(N, dtype=np.int64)
        if sequence_indices is None
        else np.asarray(sequence_indices, dtype=np.int64)
    )
    if np.any(analysis_indices < 0) or np.any(analysis_indices >= N):
        raise ValueError("sequence_indices contains indices outside X")

    X_ref = arr if sequence_indices is None else arr[analysis_indices]
    log_progress(f"codon-ism: predicting {analysis_indices.shape[0]} reference sequences", enabled=progress)
    reference_predictions = _predict(predictor, X_ref, batch_size=reference_batch_size)
    if reference_predictions.shape[0] != analysis_indices.shape[0]:
        raise ValueError("predictor must return one scalar prediction per input sequence")

    base_channel_indices, base_letters, base_to_channel, _ = _base_channels(resolved_schema)
    position_scores = (
        np.zeros((analysis_indices.shape[0], len(base_channel_indices), L), dtype=np.float32)
        if compute_position_scores
        else None
    )
    position_max_abs: dict[tuple[int, int], float] = {}
    position_base_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}

    collected_rows: list[CodonMutation] = []
    mutant_batch: list[np.ndarray] = []
    pending: list[_PendingMutation] = []
    sequence_reporter = ProgressReporter(
        "codon-ism: scan sequences",
        total=int(analysis_indices.shape[0]),
        unit="sequences",
        enabled=progress,
    )
    mutation_reporter = ProgressReporter("codon-ism: predict mutants", unit="mutations", enabled=progress)
    n_codons = 0

    def emit_rows(rows: list[CodonMutation]) -> None:
        """Collect and/or stream completed mutation rows.

        Args:
            rows: Completed codon mutation records from one prediction flush.
        """

        if not rows:
            return
        if collect:
            collected_rows.extend(rows)
        if writer is not None:
            writer.write(rows)

    def flush_mutations() -> None:
        if not mutant_batch:
            return
        batch = np.stack(mutant_batch, axis=0)
        mutant_predictions = _predict(predictor, batch, batch_size=mutation_batch_size)
        if mutant_predictions.shape[0] != len(pending):
            raise ValueError("predictor must return one scalar prediction per mutant sequence")

        rows: list[CodonMutation] = []
        for y_mut, meta in zip(mutant_predictions, pending):
            ref_pred = float(meta.reference_prediction)
            mut_pred = float(y_mut)
            delta = mut_pred - ref_pred
            rows.append(
                CodonMutation(
                    sequence_index=meta.sequence_index,
                    codon_start=meta.codon_start,
                    cds_codon_index=meta.cds_codon_index,
                    cds_start=meta.cds_start,
                    cds_end=meta.cds_end,
                    cds_length=meta.cds_length,
                    cds_relative_position=meta.cds_relative_position,
                    reference_codon=meta.reference_codon,
                    alternate_codon=meta.alternate_codon,
                    reference_amino_acid=meta.reference_amino_acid,
                    alternate_amino_acid=meta.alternate_amino_acid,
                    synonymous=meta.synonymous,
                    reference_prediction=ref_pred,
                    mutant_prediction=mut_pred,
                    delta=delta,
                )
            )
            if compute_position_scores:
                key = (meta.local_index, meta.codon_start)
                position_max_abs[key] = max(position_max_abs.get(key, 0.0), abs(float(delta)))
        emit_rows(rows)
        mutation_reporter.update(advance=len(rows))
        mutant_batch.clear()
        pending.clear()

    try:
        for local_i, seq_i_raw in enumerate(analysis_indices):
            seq_i = int(seq_i_raw)
            seq = np.asarray(arr[seq_i])
            valid_len = min(int(lengths[seq_i]), L)
            cds = find_cds_codon_starts(
                seq,
                resolved_schema,
                valid_length=valid_len,
                cds_channel=cds_channel,
            )
            if cds.cds_length < 3 or cds.starts.size == 0:
                sequence_reporter.update()
                continue

            ref_pred = float(reference_predictions[local_i])
            for codon_index, start_raw in enumerate(cds.starts):
                start = int(start_raw)
                decoded = _decode_codon(
                    seq,
                    start,
                    base_channel_indices=base_channel_indices,
                    base_letters=base_letters,
                )
                if decoded is None:
                    continue
                ref_codon, ref_base_offsets = decoded
                alts = codon_alternates(
                    ref_codon,
                    mutation_policy=mutation_policy,
                    include_stop_codons=include_stop_codons,
                )
                if not alts:
                    continue
                n_codons += 1

                if compute_position_scores:
                    position_base_offsets[(local_i, start)] = ref_base_offsets

                ref_aa = CODON_TO_AA[ref_codon]
                cds_relative_position = float((cds.cds_end - start) / cds.cds_length)
                for alt_codon in alts:
                    alt_aa = CODON_TO_AA[alt_codon]
                    mutant = seq.copy()
                    _write_codon_inplace(
                        mutant,
                        start,
                        alt_codon,
                        base_channel_indices=base_channel_indices,
                        base_to_channel=base_to_channel,
                    )
                    mutant_batch.append(mutant)
                    pending.append(
                        _PendingMutation(
                            sequence_index=seq_i,
                            local_index=local_i,
                            codon_start=start,
                            cds_codon_index=int(codon_index),
                            cds_start=cds.cds_start,
                            cds_end=cds.cds_end,
                            cds_length=cds.cds_length,
                            cds_relative_position=cds_relative_position,
                            reference_codon=ref_codon,
                            alternate_codon=alt_codon,
                            reference_amino_acid=ref_aa,
                            alternate_amino_acid=alt_aa,
                            synonymous=ref_aa == alt_aa,
                            reference_prediction=ref_pred,
                        )
                    )
                    if len(mutant_batch) >= mutation_batch_size:
                        flush_mutations()
            sequence_reporter.update()
        flush_mutations()
    finally:
        if writer is not None:
            writer.close()
    sequence_reporter.close(extra=f"{n_codons} codons scanned")
    mutation_reporter.close()

    if position_scores is not None:
        for (local_i, start), max_abs in position_max_abs.items():
            offsets = position_base_offsets[(local_i, start)]
            for j, offset in enumerate(offsets):
                position_scores[local_i, offset, start + j] = np.float32(max_abs)

    mutation_table = _rows_to_structured(collected_rows) if collect else np.empty((0,), dtype=_MUTATION_DTYPE)
    return CodonISMResult(
        mutations=mutation_table,
        reference_predictions=reference_predictions.astype(np.float32, copy=False),
        valid_lengths=lengths[analysis_indices].astype(np.int64, copy=False),
        sequence_indices=analysis_indices.astype(np.int64, copy=False),
        position_scores=position_scores,
    )


def save_codon_ism_result(
    result: CodonISMResult,
    out_dir: str | Path,
    *,
    save_mutations: bool = True,
    progress: bool = True,
) -> None:
    """Save codon ISM arrays and, optionally, the in-memory mutation table.

    Args:
        result: Codon ISM result object to serialize.
        out_dir: Destination directory for arrays and summary JSON.
        save_mutations: Whether to save in-memory mutation rows as compressed
            NPZ columns.
        progress: Whether to emit progress messages while saving.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_progress(f"codon-ism: saving results to {out}", enabled=progress)
    np.save(out / "reference_predictions.npy", result.reference_predictions)
    np.save(out / "valid_lengths.npy", result.valid_lengths)
    np.save(out / "sequence_indices.npy", result.sequence_indices)
    if result.position_scores is not None:
        np.save(out / "position_scores.npy", result.position_scores)
    if save_mutations:
        np.savez_compressed(out / "mutations.npz", **_structured_to_columns(result.mutations))
    summary = {
        "analysis": "codon_ism",
        "delta_definition": "mutant_prediction - reference_prediction",
        "n_sequences": int(result.sequence_indices.shape[0]),
        "n_mutations_in_memory": int(result.mutations.shape[0]),
        "has_position_scores": result.position_scores is not None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress("codon-ism: done", enabled=progress)
