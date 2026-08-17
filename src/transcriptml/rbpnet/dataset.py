"""Jitter-aware structured batches over materialized RBPNet bundles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from transcriptml.data.bundle import DatasetBundle


@dataclass(frozen=True)
class RBPNetBatch:
    """One structured RBPNet mini-batch."""

    sequence: torch.Tensor
    pooled_ip_profile: torch.Tensor
    sminput_profile: torch.Tensor
    individual_ip_profiles: torch.Tensor
    ip_measurement_counts: torch.Tensor
    sminput_measurement_counts: torch.Tensor
    ip_library_sizes: torch.Tensor
    sminput_library_size: torch.Tensor
    depth_offsets: torch.Tensor
    measurement_mask: torch.Tensor
    profile_valid_mask: torch.Tensor
    sequence_valid_mask: torch.Tensor
    jitter_shift: torch.Tensor
    crop_start: torch.Tensor
    selection_start: torch.Tensor
    selection_end: torch.Tensor
    indices: torch.Tensor
    example_ids: tuple[str, ...]
    replicate_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "RBPNetBatch":
        """Move tensor fields to a device while retaining identifiers."""

        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return RBPNetBatch(**values)


def _required_array(bundle: DatasetBundle, name: str) -> np.ndarray:
    try:
        return bundle.arrays[name]
    except KeyError as exc:
        raise ValueError(f"RBPNet bundle is missing named array {name!r}") from exc


def _sample_metadata(bundle: DatasetBundle) -> tuple[str, tuple[str, ...], int, np.ndarray]:
    metadata = bundle.config.get("sample_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("RBPNet bundle lacks sample_metadata")
    sminput = metadata.get("sminput")
    ips = metadata.get("ip")
    if not isinstance(sminput, Mapping) or not isinstance(ips, Sequence) or not ips:
        raise ValueError("RBPNet bundle sample_metadata must define sminput and IP samples")
    input_name = str(sminput.get("name", ""))
    input_size = int(sminput.get("effective_library_size", 0))
    ip_names = tuple(str(sample.get("name", "")) for sample in ips)
    ip_sizes = np.asarray(
        [int(sample.get("effective_library_size", 0)) for sample in ips],
        dtype=np.int64,
    )
    if not input_name or any(not name for name in ip_names):
        raise ValueError("RBPNet sample names must be non-empty")
    if input_size <= 0 or np.any(ip_sizes <= 0):
        raise ValueError("RBPNet effective library sizes must be positive")
    axis_order = tuple(str(name) for name in metadata.get("ip_axis_order", ip_names))
    if axis_order != ip_names:
        raise ValueError("RBPNet ip_axis_order disagrees with IP sample metadata")
    return input_name, ip_names, input_size, ip_sizes


class RBPNetDataset(Dataset):
    """Lazy fixed-crop view of one memory-mappable RBPNet bundle.

    Training jitter is deterministic for a given ``(seed, epoch, index)`` and
    therefore works consistently with zero or multiple DataLoader workers.
    Evaluation uses shift zero unless an explicit shift is requested through
    :meth:`item_for_shift`.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        *,
        crop_length: int | None = None,
        max_train_jitter: int = 0,
        training: bool = False,
        seed: int = 123,
        require_full_measurement_interval: bool = True,
    ) -> None:
        if bundle.config.get("bundle_format") != "transcriptml-rbpnet-bundle":
            raise ValueError("structured RBPNet training requires a TranscriptML RBPNet bundle")
        if bundle.metadata is None:
            raise ValueError("RBPNet bundle metadata is required for coordinate-aware crops")
        self.bundle = bundle
        self.metadata = bundle.metadata
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.require_full_measurement_interval = bool(require_full_measurement_interval)
        base_input_length = int(bundle.config.get("input_length", 0))
        base_profile_length = int(bundle.config.get("profile_length", 0))
        if base_input_length <= 0 or base_profile_length <= 0:
            raise ValueError("RBPNet bundle must record positive input_length and profile_length")
        if base_input_length != base_profile_length:
            raise ValueError(
                "this RBPNet family currently requires equal sequence and profile crop lengths"
            )
        self.crop_length = base_input_length if crop_length is None else int(crop_length)
        if self.crop_length != base_input_length:
            raise ValueError(
                f"crop_length {self.crop_length} must match bundle input_length {base_input_length}"
            )
        self.bundle_max_jitter = int(bundle.config.get("max_jitter", 0))
        self.max_train_jitter = int(max_train_jitter)
        if self.max_train_jitter < 0:
            raise ValueError("max_train_jitter must be non-negative")
        if self.max_train_jitter > self.bundle_max_jitter:
            raise ValueError(
                f"max_train_jitter {self.max_train_jitter} exceeds materialized margin "
                f"{self.bundle_max_jitter}"
            )
        self.boundary_policy = str(bundle.config.get("transcript_end_policy", "drop"))
        if self.boundary_policy not in {"shift_to_fit", "drop", "pad"}:
            raise ValueError(f"unsupported RBPNet boundary policy {self.boundary_policy!r}")

        self.X = bundle.X
        self.sminput_profiles = _required_array(bundle, "sminput_profiles")
        self.ip_profiles = _required_array(bundle, "ip_profiles")
        self.selection_sminput_counts = _required_array(bundle, "selection_sminput_counts")
        self.selection_ip_counts = _required_array(bundle, "selection_ip_counts")
        self.sequence_valid_masks = bundle.arrays.get("sequence_valid_mask")
        self.profile_valid_masks = bundle.arrays.get("profile_valid_mask")
        self.sminput_name, self.replicate_names, self.sminput_library_size, self.ip_library_sizes = (
            _sample_metadata(bundle)
        )
        self.depth_offsets = np.log(
            self.ip_library_sizes.astype(np.float64) / float(self.sminput_library_size)
        ).astype(np.float32)
        self._validate_shapes()
        if self.require_full_measurement_interval:
            self._validate_measurement_intervals()

    def _validate_shapes(self) -> None:
        n = int(self.X.shape[0])
        if self.X.ndim != 3 or self.X.shape[1] != 4:
            raise ValueError("RBPNet X must have shape (N, 4, materialized_length)")
        if self.sminput_profiles.ndim != 2 or self.sminput_profiles.shape[0] != n:
            raise ValueError("sminput_profiles must have shape (N, materialized_length)")
        if self.ip_profiles.ndim != 3 or self.ip_profiles.shape[:2] != (
            n, len(self.replicate_names)
        ):
            raise ValueError("ip_profiles must have shape (N, R, materialized_length)")
        expected_width = self.crop_length + 2 * self.bundle_max_jitter
        if self.X.shape[-1] != expected_width or self.sminput_profiles.shape[-1] != expected_width:
            raise ValueError("materialized sequence/profile width disagrees with bundle jitter contract")
        if self.ip_profiles.shape[-1] != expected_width:
            raise ValueError("IP profile width disagrees with sequence/profile width")
        if self.selection_sminput_counts.shape != (n,):
            raise ValueError("selection_sminput_counts must have shape (N,)")
        if self.selection_ip_counts.shape != (n, len(self.replicate_names)):
            raise ValueError("selection_ip_counts must have shape (N, R)")
        for name, masks in (
            ("sequence_valid_mask", self.sequence_valid_masks),
            ("profile_valid_mask", self.profile_valid_masks),
        ):
            if masks is not None and masks.shape != (n, expected_width):
                raise ValueError(f"{name} must have shape (N, materialized_length)")

    def _metadata_coordinates(self, index: int) -> tuple[int, int, int, int, int, int, int]:
        row = self.metadata[int(index)]
        required = ("transcript_anchor", "selection_start", "selection_end", "locus_length")
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(
                f"RBPNet metadata row {index} lacks coordinate fields: {', '.join(missing)}"
            )
        sequence_start = int(
            row.get("sequence_materialized_start", row.get("sequence_context_start"))
        )
        profile_start = int(
            row.get("profile_materialized_start", row.get("profile_context_start"))
        )
        return (
            int(row["transcript_anchor"]),
            int(row["selection_start"]),
            int(row["selection_end"]),
            int(row["locus_length"]),
            sequence_start,
            profile_start,
            self.crop_length,
        )

    def _crop_offsets(self, index: int, jitter_shift: int) -> tuple[int, int, int]:
        anchor, _, _, locus_length, sequence_start, profile_start, length = (
            self._metadata_coordinates(index)
        )
        shift = int(jitter_shift)
        if abs(shift) > self.max_train_jitter:
            raise ValueError(
                f"requested jitter shift {shift} exceeds configured range "
                f"[-{self.max_train_jitter}, {self.max_train_jitter}]"
            )
        if self.boundary_policy == "pad":
            sequence_offset = self.bundle_max_jitter + shift
            profile_offset = self.bundle_max_jitter + shift
            crop_start = sequence_start + sequence_offset
        else:
            if locus_length < length:
                raise ValueError("locus is shorter than the requested RBPNet crop")
            desired_start = anchor - length // 2 + shift
            crop_start = min(max(desired_start, 0), locus_length - length)
            sequence_offset = crop_start - sequence_start
            profile_offset = crop_start - profile_start
        if sequence_offset < 0 or sequence_offset + length > self.X.shape[-1]:
            raise ValueError("coordinate-derived sequence crop lies outside materialized context")
        if profile_offset < 0 or profile_offset + length > self.sminput_profiles.shape[-1]:
            raise ValueError("coordinate-derived profile crop lies outside materialized context")
        profile_crop_start = profile_start + profile_offset
        if profile_crop_start != crop_start:
            raise ValueError("sequence and profile crops do not describe the same biological interval")
        return sequence_offset, profile_offset, crop_start

    def _measurement_mask(self, index: int, crop_start: int) -> np.ndarray:
        _, selection_start, selection_end, _, _, _, length = self._metadata_coordinates(index)
        if selection_end <= selection_start:
            raise ValueError(f"RBPNet example {index} has an empty selection interval")
        crop_end = crop_start + length
        overlap_start = max(selection_start, crop_start)
        overlap_end = min(selection_end, crop_end)
        fully_contained = selection_start >= crop_start and selection_end <= crop_end
        if self.require_full_measurement_interval and not fully_contained:
            raise ValueError(
                f"selection interval {selection_start}-{selection_end} for example "
                f"{self.bundle.ids[index]} is not fully contained in jittered crop "
                f"{crop_start}-{crop_end}; reduce jitter or use a larger model context"
            )
        mask = np.zeros(length, dtype=np.float32)
        if overlap_end > overlap_start:
            mask[overlap_start - crop_start : overlap_end - crop_start] = 1.0
        if not np.any(mask):
            raise ValueError(f"selection interval for example {self.bundle.ids[index]} misses model crop")
        return mask

    def _validate_measurement_intervals(self) -> None:
        shifts = {-self.max_train_jitter, self.max_train_jitter, 0}
        for index in range(len(self)):
            for shift in shifts:
                _, _, crop_start = self._crop_offsets(index, shift)
                self._measurement_mask(index, crop_start)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic training-jitter epoch."""

        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _sample_shift(self, index: int) -> int:
        if not self.training or self.max_train_jitter == 0:
            return 0
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        return int(rng.integers(-self.max_train_jitter, self.max_train_jitter + 1))

    def item_for_shift(self, index: int, jitter_shift: int) -> dict[str, object]:
        """Return one example using an explicit jitter shift, useful for testing."""

        i = int(index)
        sequence_offset, profile_offset, crop_start = self._crop_offsets(i, jitter_shift)
        end_sequence = sequence_offset + self.crop_length
        end_profile = profile_offset + self.crop_length
        sequence = np.asarray(self.X[i, :, sequence_offset:end_sequence], dtype=np.float32)
        individual_ip = np.asarray(
            self.ip_profiles[i, :, profile_offset:end_profile], dtype=np.float32
        )
        sminput = np.asarray(
            self.sminput_profiles[i, profile_offset:end_profile], dtype=np.float32
        )
        sequence_valid = (
            np.ones(self.crop_length, dtype=bool)
            if self.sequence_valid_masks is None
            else np.asarray(
                self.sequence_valid_masks[i, sequence_offset:end_sequence], dtype=bool
            )
        )
        profile_valid = (
            np.ones(self.crop_length, dtype=bool)
            if self.profile_valid_masks is None
            else np.asarray(
                self.profile_valid_masks[i, profile_offset:end_profile], dtype=bool
            )
        )
        return {
            "sequence": sequence,
            "pooled_ip_profile": individual_ip.sum(axis=0, dtype=np.float32),
            "sminput_profile": sminput,
            "individual_ip_profiles": individual_ip,
            "ip_measurement_counts": np.asarray(
                self.selection_ip_counts[i], dtype=np.float32
            ),
            "sminput_measurement_counts": np.float32(self.selection_sminput_counts[i]),
            "ip_library_sizes": self.ip_library_sizes.astype(np.float32, copy=False),
            "sminput_library_size": np.float32(self.sminput_library_size),
            "depth_offsets": self.depth_offsets,
            "measurement_mask": self._measurement_mask(i, crop_start),
            "profile_valid_mask": profile_valid,
            "sequence_valid_mask": sequence_valid,
            "jitter_shift": np.int64(jitter_shift),
            "crop_start": np.int64(crop_start),
            "selection_start": np.int64(self.metadata[i]["selection_start"]),
            "selection_end": np.int64(self.metadata[i]["selection_end"]),
            "index": np.int64(i),
            "example_id": str(self.bundle.ids[i]),
            "replicate_names": self.replicate_names,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.item_for_shift(int(index), self._sample_shift(int(index)))


def collate_rbpnet(batch: list[Mapping[str, object]]) -> RBPNetBatch:
    """Stack structured examples without obscuring replicate/sample axes."""

    if not batch:
        raise ValueError("cannot collate an empty RBPNet batch")
    replicate_names = tuple(batch[0]["replicate_names"])
    if any(tuple(item["replicate_names"]) != replicate_names for item in batch):
        raise ValueError("RBPNet batch mixes incompatible replicate axes")

    def stack(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.stack([np.asarray(item[name]) for item in batch]), dtype=dtype)

    return RBPNetBatch(
        sequence=stack("sequence"),
        pooled_ip_profile=stack("pooled_ip_profile"),
        sminput_profile=stack("sminput_profile"),
        individual_ip_profiles=stack("individual_ip_profiles"),
        ip_measurement_counts=stack("ip_measurement_counts"),
        sminput_measurement_counts=stack("sminput_measurement_counts"),
        ip_library_sizes=stack("ip_library_sizes"),
        sminput_library_size=stack("sminput_library_size"),
        depth_offsets=stack("depth_offsets"),
        measurement_mask=stack("measurement_mask"),
        profile_valid_mask=stack("profile_valid_mask", dtype=torch.bool),
        sequence_valid_mask=stack("sequence_valid_mask", dtype=torch.bool),
        jitter_shift=stack("jitter_shift", dtype=torch.long).reshape(-1),
        crop_start=stack("crop_start", dtype=torch.long).reshape(-1),
        selection_start=stack("selection_start", dtype=torch.long).reshape(-1),
        selection_end=stack("selection_end", dtype=torch.long).reshape(-1),
        indices=stack("index", dtype=torch.long).reshape(-1),
        example_ids=tuple(str(item["example_id"]) for item in batch),
        replicate_names=replicate_names,
    )


def deduplicate_locus_indices(
    bundle: DatasetBundle,
    indices: Sequence[int],
) -> tuple[list[int], int]:
    """Remove replicate-eligibility duplicate rows while retaining all R tracks."""

    if bundle.metadata is None:
        raise ValueError("cannot deduplicate RBPNet loci without bundle metadata")
    kept: list[int] = []
    first_by_key: dict[tuple[object, ...], int] = {}
    for raw_index in indices:
        index = int(raw_index)
        row = bundle.metadata[index]
        key = (
            row.get("coordinate_space", bundle.config.get("coordinate_space")),
            row.get("transcript_id"),
            int(row.get("selection_start", -1)),
            int(row.get("selection_end", -1)),
            int(row.get("transcript_anchor", -1)),
        )
        if key not in first_by_key:
            first_by_key[key] = index
            kept.append(index)
    return kept, len(indices) - len(kept)
