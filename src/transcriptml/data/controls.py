from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from transcriptml.data.bundle import DatasetBundle, save_bundle_metadata
from transcriptml.data.schemas import SequenceSchema, get_schema
from transcriptml.interpret.codon_ism import CDSCodonStarts, find_cds_codon_starts
from transcriptml.progress import ProgressReporter, log_progress

RegionName = Literal["5utr", "cds", "3utr", "transcript"]
OperationName = Literal["shuffle_nucleotides", "shuffle_codons", "randomize_nucleotides", "cds_frameshift"]

_ANNOTATED_REGIONS: tuple[RegionName, ...] = ("5utr", "cds", "3utr")
_ALL_REGIONS_BY_OPERATION: dict[OperationName, tuple[RegionName, ...]] = {
    "shuffle_nucleotides": _ANNOTATED_REGIONS,
    "shuffle_codons": ("cds",),
    "randomize_nucleotides": _ANNOTATED_REGIONS,
    "cds_frameshift": ("cds",),
}

_OPERATION_ALIASES = {
    "shuffle_nucleotides": "shuffle_nucleotides",
    "shuffle_nt": "shuffle_nucleotides",
    "nucleotide_shuffle": "shuffle_nucleotides",
    "permute_nucleotides": "shuffle_nucleotides",
    "permute_nt": "shuffle_nucleotides",
    "true_scramble": "shuffle_nucleotides",
    "shuffle_codons": "shuffle_codons",
    "shuffle_codon": "shuffle_codons",
    "codon_shuffle": "shuffle_codons",
    "permute_codons": "shuffle_codons",
    "permute_codon": "shuffle_codons",
    "randomize_nucleotides": "randomize_nucleotides",
    "randomize_nt": "randomize_nucleotides",
    "random_nucleotides": "randomize_nucleotides",
    "random_nt": "randomize_nucleotides",
    "replace_random": "randomize_nucleotides",
    "ablate": "randomize_nucleotides",
    "cds_frameshift": "cds_frameshift",
    "cds_frame_shift": "cds_frameshift",
    "frameshift": "cds_frameshift",
    "frame_shift": "cds_frameshift",
}

_REGION_ALIASES: dict[str, RegionName | tuple[RegionName, ...]] = {
    "5utr": "5utr",
    "utr5": "5utr",
    "5putr": "5utr",
    "fiveutr": "5utr",
    "fiveprimeutr": "5utr",
    "5primeutr": "5utr",
    "cds": "cds",
    "coding": "cds",
    "codingsequence": "cds",
    "orf": "cds",
    "3utr": "3utr",
    "utr3": "3utr",
    "3putr": "3utr",
    "threeutr": "3utr",
    "threeprimeutr": "3utr",
    "3primeutr": "3utr",
    "transcript": "transcript",
    "wholetranscript": "transcript",
    "fulltranscript": "transcript",
    "sequence": "transcript",
}

_LEGACY_REGION_KEYS = {
    "5putrablation": "5utr",
    "5utrablation": "5utr",
    "utr5ablation": "5utr",
    "cdsablation": "cds",
    "3putrablation": "3utr",
    "3utrablation": "3utr",
    "utr3ablation": "3utr",
}

_REGION_SEED_OFFSETS = {
    "5utr": 0x9E3779B97F4A7C15,
    "cds": 0xC2B2AE3D27D4EB4F,
    "3utr": 0x165667B19E3779F9,
    "transcript": 0x85EBCA77C2B2AE63,
}

_OPERATION_SEED_OFFSETS = {
    "shuffle_nucleotides": 0x27D4EB2F165667C5,
    "shuffle_codons": 0xA24BAED4963EE407,
    "randomize_nucleotides": 0x94D049BB133111EB,
}


@dataclass(frozen=True)
class SequenceControlOperation:
    """One sequence-control operation applied to one or more regions."""

    operation: OperationName
    regions: tuple[RegionName, ...]
    shift: int | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"operation": self.operation, "regions": list(self.regions)}
        if self.operation == "cds_frameshift":
            out["shift"] = self.shift
        return out


@dataclass(frozen=True)
class SequenceControlConfig:
    """Normalized training-time RNA sequence-control configuration."""

    operations: tuple[SequenceControlOperation, ...] = ()
    seed: int = 0
    save_dir: str | None = None
    save: bool = False
    cds_channel: str | int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "seed": int(self.seed),
            "save_dir": self.save_dir,
            "save": bool(self.save),
            "cds_channel": self.cds_channel,
        }


def _control_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace("'", "")
        .replace('"', "")
        .replace(" ", "")
    )


def _operation_name(value: str) -> OperationName:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _OPERATION_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            "sequence control operation must be one of: "
            "shuffle_nucleotides, shuffle_codons, randomize_nucleotides, cds_frameshift"
        ) from exc


def _dedupe_regions(regions: Sequence[RegionName]) -> tuple[RegionName, ...]:
    seen: set[RegionName] = set()
    out: list[RegionName] = []
    for region in regions:
        if region not in seen:
            out.append(region)
            seen.add(region)
    return tuple(out)


def _normalize_region_token(value: str, *, all_regions: Sequence[RegionName]) -> tuple[RegionName, ...]:
    key = _control_key(value)
    if key in {"all", "allregions", "annotated", "annotatedregions"}:
        return tuple(all_regions)
    if key in {"", "none", "false", "off", "no"}:
        return ()
    try:
        region = _REGION_ALIASES[key]
    except KeyError as exc:
        raise ValueError("regions must be drawn from: 5utr, cds, 3utr, all, transcript") from exc
    if isinstance(region, tuple):
        return region
    return (region,)


def _regions_from_value(
    value: object,
    *,
    default_regions: Sequence[RegionName],
    all_regions: Sequence[RegionName],
) -> tuple[RegionName, ...]:
    if value is None or value is False:
        return ()
    if value is True:
        return tuple(default_regions)
    if isinstance(value, str):
        regions: list[RegionName] = []
        for token in value.replace(";", ",").replace("|", ",").split(","):
            regions.extend(_normalize_region_token(token, all_regions=all_regions))
        return _dedupe_regions(regions)
    if isinstance(value, MappingABC):
        enabled = value.get("enabled", True)
        if enabled is False:
            return ()
        if "regions" in value:
            return _regions_from_value(
                value["regions"],
                default_regions=default_regions,
                all_regions=all_regions,
            )
        if "region" in value:
            return _regions_from_value(
                value["region"],
                default_regions=default_regions,
                all_regions=all_regions,
            )
        return tuple(default_regions)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        regions = []
        for item in value:
            regions.extend(
                _regions_from_value(
                    item,
                    default_regions=default_regions,
                    all_regions=all_regions,
                )
            )
        return _dedupe_regions(regions)
    raise TypeError(f"Unsupported regions value: {value!r}")


def _frameshift_from_value(value: object) -> int:
    if isinstance(value, MappingABC):
        for key in ("shift", "frameshift", "frame_shift", "amount"):
            if key in value:
                return _frameshift_from_value(value[key])
        raise ValueError("cds_frameshift requires a shift of 1 or 2")
    if value is None or isinstance(value, bool):
        raise ValueError("cds_frameshift requires a shift of 1 or 2")
    try:
        shift = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cds_frameshift requires a shift of 1 or 2") from exc
    if shift not in {1, 2}:
        raise ValueError("cds_frameshift shift must be 1 or 2")
    return shift


def _operation_from_entry(entry: Mapping[str, object]) -> SequenceControlOperation:
    raw_operation = entry.get("operation", entry.get("type", entry.get("op")))
    if raw_operation is None:
        raise ValueError("Each sequence control operation needs an 'operation' or 'type' field")
    operation = _operation_name(str(raw_operation))
    shift = None
    if operation == "cds_frameshift":
        shift = _frameshift_from_value(
            entry.get("shift", entry.get("frameshift", entry.get("frame_shift", entry.get("amount"))))
        )
    regions = _regions_from_value(
        entry.get("regions", entry.get("region", True)),
        default_regions=_ALL_REGIONS_BY_OPERATION[operation],
        all_regions=_ALL_REGIONS_BY_OPERATION[operation],
    )
    return SequenceControlOperation(operation=operation, regions=regions, shift=shift)


def _legacy_operation(region: RegionName, mode: object) -> SequenceControlOperation | None:
    if mode is None:
        return None
    key = _control_key(str(mode))
    if key in {"", "none", "false", "off", "no"}:
        return None
    if key == "scramble":
        operation: OperationName = "shuffle_codons" if region == "cds" else "shuffle_nucleotides"
    elif key == "truescramble":
        operation = "shuffle_nucleotides"
    elif key == "ablate":
        operation = "randomize_nucleotides"
    elif key == "ablatesavemotif":
        raise ValueError("Legacy ablate_savemotif is not supported by sequence_controls")
    else:
        raise ValueError(f"Unsupported legacy ablation mode: {mode!r}")
    return SequenceControlOperation(operation=operation, regions=(region,))


def _merge_operations(operations: Sequence[SequenceControlOperation]) -> tuple[SequenceControlOperation, ...]:
    merged: dict[OperationName, SequenceControlOperation] = {}
    for operation in operations:
        if not operation.regions:
            continue
        if operation.operation in {"shuffle_codons", "cds_frameshift"} and any(
            region != "cds" for region in operation.regions
        ):
            raise ValueError(f"{operation.operation} only supports the CDS region")
        existing = merged.get(operation.operation)
        if existing is not None and operation.operation == "cds_frameshift" and existing.shift != operation.shift:
            raise ValueError("Duplicate cds_frameshift operations must use the same shift")
        bucket = list(existing.regions) if existing is not None else []
        for region in operation.regions:
            if region not in bucket:
                bucket.append(region)
        merged[operation.operation] = SequenceControlOperation(
            operation=operation.operation,
            regions=tuple(bucket),
            shift=operation.shift if operation.operation == "cds_frameshift" else None,
        )
    out = tuple(merged.values())
    owners: dict[RegionName, OperationName] = {}
    for operation in out:
        if operation.operation == "cds_frameshift":
            continue
        for region in operation.regions:
            if region == "transcript" and owners:
                raise ValueError("The transcript region cannot be combined with other sequence-control regions")
            if owners.get("transcript") is not None and region != "transcript":
                raise ValueError("The transcript region cannot be combined with other sequence-control regions")
            previous = owners.get(region)
            if previous is not None and previous != operation.operation:
                raise ValueError(
                    f"Region '{region}' is targeted by both {previous} and {operation.operation}; "
                    "use at most one sequence-control operation per region"
                )
            owners[region] = operation.operation
    return out


def normalize_sequence_control_config(config: object) -> SequenceControlConfig:
    """Normalize user-facing sequence-control config into explicit operations.

    The preferred shape is::

        {
          "seed": 42,
          "operations": [
            {"operation": "shuffle_nucleotides", "regions": ["5utr", "3utr"]},
            {"operation": "shuffle_codons", "regions": ["cds"]},
            {"operation": "cds_frameshift", "shift": 1}
          ],
          "save_dir": "data/saluki_control"
        }

    Top-level shortcuts such as ``"shuffle_nucleotides": ["5utr"]`` and the
    older ``5pUTR_ablation``/``CDS_ablation``/``3pUTR_ablation`` names are also
    accepted.
    """

    if config is None or config is False:
        return SequenceControlConfig()
    if isinstance(config, SequenceControlConfig):
        return config

    entries: list[SequenceControlOperation] = []
    seed = 0
    save = False
    save_dir: str | None = None
    cds_channel: str | int | None = None

    if isinstance(config, SequenceABC) and not isinstance(config, (str, bytes, bytearray)):
        for entry in config:
            if not isinstance(entry, MappingABC):
                raise TypeError("sequence_controls list entries must be mappings")
            entries.append(_operation_from_entry(entry))
        return SequenceControlConfig(operations=_merge_operations(entries))

    if not isinstance(config, MappingABC):
        raise TypeError("sequence_controls must be a mapping, list of operations, or null")

    if config.get("enabled", True) is False:
        return SequenceControlConfig()

    seed = int(config.get("seed", config.get("ablation_seed", 0)))
    save = bool(config.get("save", False))
    if config.get("save_dir") is not None:
        save_dir = str(config["save_dir"])
        save = True
    cds_channel = config.get("cds_channel")
    if isinstance(cds_channel, str) and cds_channel.isdigit():
        cds_channel = int(cds_channel)

    if any(key in config for key in ("operation", "type", "op")):
        entries.append(_operation_from_entry(config))

    raw_operations = config.get("operations")
    if raw_operations is not None:
        if not isinstance(raw_operations, SequenceABC) or isinstance(raw_operations, (str, bytes, bytearray)):
            raise TypeError("sequence_controls.operations must be a list")
        for entry in raw_operations:
            if not isinstance(entry, MappingABC):
                raise TypeError("sequence_controls.operations entries must be mappings")
            entries.append(_operation_from_entry(entry))

    shortcut_keys: tuple[tuple[str, OperationName], ...] = (
        ("shuffle_nucleotides", "shuffle_nucleotides"),
        ("shuffle_nt", "shuffle_nucleotides"),
        ("permute_nucleotides", "shuffle_nucleotides"),
        ("shuffle_codons", "shuffle_codons"),
        ("codon_shuffle", "shuffle_codons"),
        ("permute_codons", "shuffle_codons"),
        ("randomize_nucleotides", "randomize_nucleotides"),
        ("randomize_nt", "randomize_nucleotides"),
        ("random_nucleotides", "randomize_nucleotides"),
        ("replace_random", "randomize_nucleotides"),
        ("cds_frameshift", "cds_frameshift"),
        ("cds_frame_shift", "cds_frameshift"),
    )
    for key, operation in shortcut_keys:
        if key not in config:
            continue
        if operation == "cds_frameshift":
            value = config[key]
            regions = _regions_from_value(
                value.get("regions", value.get("region", True)) if isinstance(value, MappingABC) else True,
                default_regions=_ALL_REGIONS_BY_OPERATION[operation],
                all_regions=_ALL_REGIONS_BY_OPERATION[operation],
            )
            entries.append(
                SequenceControlOperation(
                    operation=operation,
                    regions=regions,
                    shift=_frameshift_from_value(value),
                )
            )
            continue
        regions = _regions_from_value(
            config[key],
            default_regions=_ALL_REGIONS_BY_OPERATION[operation],
            all_regions=_ALL_REGIONS_BY_OPERATION[operation],
        )
        entries.append(SequenceControlOperation(operation=operation, regions=regions))

    for raw_key, value in config.items():
        region = _LEGACY_REGION_KEYS.get(_control_key(str(raw_key)))
        if region is None:
            continue
        legacy = _legacy_operation(region, value)
        if legacy is not None:
            entries.append(legacy)

    return SequenceControlConfig(
        operations=_merge_operations(entries),
        seed=seed,
        save_dir=save_dir,
        save=save,
        cds_channel=cds_channel,
    )


def _base_channel_indices(schema: SequenceSchema) -> np.ndarray:
    indices = []
    letters = []
    for base_name in schema.base_channels:
        if base_name not in schema.channels:
            raise ValueError(f"Base channel '{base_name}' is not present in schema channels {schema.channels}")
        letter = base_name.upper().replace("T", "U")
        if letter not in {"A", "C", "G", "U"}:
            raise ValueError(f"Unsupported base channel '{base_name}'; expected A/C/G/U/T")
        indices.append(schema.channels.index(base_name))
        letters.append(letter)
    if set(letters) != {"A", "C", "G", "U"} or len(letters) != 4:
        raise ValueError("sequence_controls requires exactly one A, C, G, and U/T base channel")
    return np.asarray(indices, dtype=np.int64)


def _resolve_cds_channel(schema: SequenceSchema, cds_channel: str | int | None) -> int:
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


def _infer_valid_length(x: np.ndarray, base_channels: np.ndarray) -> int:
    base = np.asarray(x[base_channels])
    nonzero = np.any(base != 0, axis=0)
    idx = np.nonzero(nonzero)[0]
    return int(idx[-1] + 1) if idx.size else 0


def _mixed_rng(seed: int, seq_index: int, operation: OperationName, region: RegionName) -> np.random.Generator:
    value = (
        (int(seed) & 0xFFFFFFFFFFFFFFFF)
        ^ (((int(seq_index) + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        ^ _OPERATION_SEED_OFFSETS[operation]
        ^ _REGION_SEED_OFFSETS[region]
    )
    value ^= (value >> 30) & 0xFFFFFFFFFFFFFFFF
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= (value >> 27) & 0xFFFFFFFFFFFFFFFF
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= (value >> 31) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(int(value))


def _base_symbols(x: np.ndarray, start: int, end: int, base_channels: np.ndarray) -> np.ndarray:
    if end <= start:
        return np.empty((0,), dtype=np.int16)
    region = np.asarray(x[base_channels, int(start) : int(end)])
    called = np.count_nonzero(region, axis=0) == 1
    symbols = np.full(region.shape[1], -1, dtype=np.int16)
    if np.any(called):
        symbols[called] = np.argmax(region[:, called], axis=0).astype(np.int16, copy=False)
    return symbols


def _write_base_symbols(
    x: np.ndarray,
    start: int,
    end: int,
    symbols: np.ndarray,
    base_channels: np.ndarray,
) -> None:
    if end <= start:
        return
    start = int(start)
    end = int(end)
    x[base_channels, start:end] = 0
    valid = np.asarray(symbols) >= 0
    if not np.any(valid):
        return
    cols = start + np.nonzero(valid)[0]
    channel_offsets = np.asarray(symbols[valid], dtype=np.int64)
    x[base_channels[channel_offsets], cols] = 1


def _region_bounds(
    region: RegionName,
    *,
    valid_length: int,
    cds: CDSCodonStarts | None,
) -> tuple[int, int] | None:
    if region == "transcript":
        return 0, int(valid_length)
    if cds is None or cds.cds_length < 3 or cds.starts.size == 0:
        return None
    cds_start = max(0, int(cds.cds_start))
    cds_end = min(int(valid_length), int(cds.cds_end) + 1)
    if region == "5utr":
        return 0, cds_start
    if region == "cds":
        return cds_start, cds_end
    return cds_end, int(valid_length)


def _shuffle_nucleotides(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    symbols = _base_symbols(x, start, end, base_channels)
    if symbols.size > 1:
        symbols = symbols[rng.permutation(symbols.size)]
    _write_base_symbols(x, start, end, symbols, base_channels)


def _randomize_nucleotides(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    length = max(0, int(end) - int(start))
    symbols = rng.integers(0, int(base_channels.size), size=length, dtype=np.int16)
    _write_base_symbols(x, start, end, symbols, base_channels)


def _frameshift_cds_channel(
    x: np.ndarray,
    *,
    valid_length: int,
    cds_channel: int,
    shift: int,
) -> bool:
    if valid_length <= 0:
        return False
    channel = int(cds_channel)
    end = min(int(valid_length), int(x.shape[-1]))
    locs = np.flatnonzero(np.asarray(x[channel, :end]) > 0)
    if locs.size == 0:
        return False
    shifted = locs + int(shift)
    shifted = shifted[shifted < end]
    x[channel, :end] = 0
    if shifted.size:
        x[channel, shifted] = 1
    return True


def _shuffle_codons(
    x: np.ndarray,
    *,
    cds: CDSCodonStarts,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    starts = np.asarray(cds.starts, dtype=np.int64)
    starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
    if starts.size == 0:
        return
    codons = np.stack([_base_symbols(x, int(start), int(start) + 3, base_channels) for start in starts], axis=0)
    if codons.shape[0] > 1:
        codons = codons[rng.permutation(codons.shape[0])]
    for start, codon in zip(starts.tolist(), codons, strict=True):
        _write_base_symbols(x, int(start), int(start) + 3, codon, base_channels)


def _empty_nested_counts() -> dict[str, dict[str, int]]:
    return {
        "shuffle_nucleotides": {region: 0 for region in _ANNOTATED_REGIONS + ("transcript",)},
        "shuffle_codons": {region: 0 for region in _ANNOTATED_REGIONS + ("transcript",)},
        "randomize_nucleotides": {region: 0 for region in _ANNOTATED_REGIONS + ("transcript",)},
        "cds_frameshift": {region: 0 for region in _ANNOTATED_REGIONS + ("transcript",)},
    }


def _stats(config: SequenceControlConfig, n_sequences: int) -> dict[str, object]:
    return {
        "n_sequences": int(n_sequences),
        "config": config.to_dict(),
        "edited": _empty_nested_counts(),
        "skipped_empty_region": _empty_nested_counts(),
        "skipped_no_cds": 0,
        "skipped_empty_transcript": 0,
    }


def _increment(stats: dict[str, object], group: str, operation: OperationName, region: RegionName) -> None:
    nested = stats[group]
    assert isinstance(nested, dict)
    by_region = nested[operation]
    assert isinstance(by_region, dict)
    by_region[region] = int(by_region.get(region, 0)) + 1


def apply_sequence_controls_array(
    X: np.ndarray,
    config: object,
    *,
    schema: str | SequenceSchema = "saluki6",
    out_path: str | Path | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply RNA sequence controls to an encoded array.

    Base perturbations rewrite only base channels. ``cds_frameshift`` rewrites
    only the CDS/codon-start channel. The splice-junction channel is copied
    through unchanged.
    """

    cfg = normalize_sequence_control_config(config)
    arr = np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    stats = _stats(cfg, int(arr.shape[0]))
    if not cfg.operations:
        return arr, stats

    resolved = get_schema(schema)
    if arr.shape[1] < resolved.n_channels:
        raise ValueError(
            f"X has {arr.shape[1]} channels, but schema '{resolved.name}' expects {resolved.n_channels}"
        )
    base_channels = _base_channel_indices(resolved)
    needs_frameshift = any(op.operation == "cds_frameshift" for op in cfg.operations)
    cds_channel_index = _resolve_cds_channel(resolved, cfg.cds_channel) if needs_frameshift else None

    if out_path is None:
        out = np.array(arr, copy=True)
    else:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)

    needs_cds = any(
        op.operation != "cds_frameshift" and region != "transcript"
        for op in cfg.operations
        for region in op.regions
    )
    reporter = ProgressReporter(
        "sequence-controls: edit transcripts",
        total=int(arr.shape[0]),
        unit="transcripts",
        enabled=progress,
    )
    for seq_index in range(int(arr.shape[0])):
        if out_path is not None:
            out[seq_index] = arr[seq_index]
        x_one = out[seq_index]
        valid_length = _infer_valid_length(x_one, base_channels)
        valid_length = min(valid_length, int(arr.shape[-1]))
        if valid_length <= 0:
            stats["skipped_empty_transcript"] = int(stats["skipped_empty_transcript"]) + 1
            reporter.update()
            continue

        cds: CDSCodonStarts | None = None
        if needs_cds:
            try:
                cds = find_cds_codon_starts(
                    x_one,
                    resolved,
                    valid_length=valid_length,
                    cds_channel=cfg.cds_channel,
                )
            except ValueError as exc:
                raise ValueError(
                    "sequence_controls requires a resolvable CDS channel for 5utr/cds/3utr regions"
                ) from exc
            if cds.starts.size == 0 or cds.cds_length < 3:
                stats["skipped_no_cds"] = int(stats["skipped_no_cds"]) + 1
        elif needs_frameshift and cds_channel_index is not None:
            if not np.any(np.asarray(x_one[cds_channel_index, :valid_length]) > 0):
                stats["skipped_no_cds"] = int(stats["skipped_no_cds"]) + 1

        for op in cfg.operations:
            if op.operation == "cds_frameshift":
                if cds_channel_index is None:
                    continue
                if op.shift is None:
                    raise ValueError("cds_frameshift requires a shift of 1 or 2")
                shifted = _frameshift_cds_channel(
                    x_one,
                    valid_length=valid_length,
                    cds_channel=cds_channel_index,
                    shift=op.shift,
                )
                if shifted:
                    _increment(stats, "edited", op.operation, "cds")
                continue
            for region in op.regions:
                bounds = _region_bounds(region, valid_length=valid_length, cds=cds)
                if bounds is None:
                    continue
                start, end = bounds
                if end <= start:
                    _increment(stats, "skipped_empty_region", op.operation, region)
                    continue
                rng = _mixed_rng(cfg.seed, seq_index, op.operation, region)
                if op.operation == "shuffle_nucleotides":
                    _shuffle_nucleotides(x_one, start=start, end=end, base_channels=base_channels, rng=rng)
                elif op.operation == "randomize_nucleotides":
                    _randomize_nucleotides(x_one, start=start, end=end, base_channels=base_channels, rng=rng)
                elif op.operation == "shuffle_codons":
                    if cds is None:
                        continue
                    _shuffle_codons(x_one, cds=cds, base_channels=base_channels, rng=rng)
                else:  # pragma: no cover - guarded by normalization.
                    raise ValueError(f"Unsupported sequence-control operation: {op.operation}")
                _increment(stats, "edited", op.operation, region)
        reporter.update()
    if hasattr(out, "flush"):
        out.flush()
    reporter.close()
    return out, stats


def apply_sequence_controls_to_bundle(
    bundle: DatasetBundle,
    config: object,
    *,
    default_save_dir: str | Path | None = None,
    progress: bool = True,
) -> tuple[DatasetBundle, dict[str, object]]:
    """Apply sequence controls to a dataset bundle and optionally save it."""

    cfg = normalize_sequence_control_config(config)
    if not cfg.operations:
        return bundle, _stats(cfg, int(bundle.X.shape[0]))

    save_dir: Path | None = None
    if cfg.save_dir is not None:
        save_dir = Path(cfg.save_dir)
    elif cfg.save and default_save_dir is not None:
        save_dir = Path(default_save_dir)

    log_progress(
        "sequence-controls: "
        + ", ".join(f"{op.operation}({','.join(op.regions)})" for op in cfg.operations),
        enabled=progress,
    )

    out_path = save_dir / "X.npy" if save_dir is not None else None
    X_controlled, stats = apply_sequence_controls_array(
        bundle.X,
        cfg,
        schema=bundle.schema,
        out_path=out_path,
        progress=progress,
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if bundle.y is not None:
            np.save(save_dir / "y.npy", np.asarray(bundle.y))
        stats["save_dir"] = str(save_dir)

    controlled_config = dict(bundle.config)
    controlled_config["sequence_controls"] = cfg.to_dict()
    controlled_config["sequence_control_stats"] = stats
    controlled = DatasetBundle(
        X=X_controlled,
        y=bundle.y,
        ids=bundle.ids,
        schema=bundle.schema,
        metadata=bundle.metadata,
        splits=bundle.splits,
        config=controlled_config,
    )
    if save_dir is not None:
        save_bundle_metadata(controlled, save_dir)
        (save_dir / "sequence_controls.json").write_text(
            json.dumps({"config": cfg.to_dict(), "stats": stats}, indent=2),
            encoding="utf-8",
        )
        log_progress(f"sequence-controls: saved controlled bundle to {save_dir}", enabled=progress)
    return controlled, stats
