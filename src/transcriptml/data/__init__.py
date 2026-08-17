"""Data processing utilities, exposed lazily to keep assay CLIs lightweight."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DatasetBundle": ("transcriptml.data.bundle", "DatasetBundle"),
    "load_bundle": ("transcriptml.data.bundle", "load_bundle"),
    "save_bundle": ("transcriptml.data.bundle", "save_bundle"),
    "SequenceControlConfig": ("transcriptml.data.controls", "SequenceControlConfig"),
    "SequenceControlOperation": ("transcriptml.data.controls", "SequenceControlOperation"),
    "apply_sequence_controls_array": ("transcriptml.data.controls", "apply_sequence_controls_array"),
    "apply_sequence_controls_to_bundle": ("transcriptml.data.controls", "apply_sequence_controls_to_bundle"),
    "normalize_sequence_control_config": ("transcriptml.data.controls", "normalize_sequence_control_config"),
    "DEFAULT_SALUKI_LENGTH": ("transcriptml.data.encoding", "DEFAULT_SALUKI_LENGTH"),
    "encode_rna_sequence": ("transcriptml.data.encoding", "encode_rna_sequence"),
    "encode_saluki_transcript": ("transcriptml.data.encoding", "encode_saluki_transcript"),
    "encode_sequences": ("transcriptml.data.encoding", "encode_sequences"),
    "infer_valid_length": ("transcriptml.data.encoding", "infer_valid_length"),
    "infer_valid_lengths": ("transcriptml.data.encoding", "infer_valid_lengths"),
    "TranscriptFeature": ("transcriptml.data.genomics", "TranscriptFeature"),
    "TranscriptRecord": ("transcriptml.data.genomics", "TranscriptRecord"),
    "extract_transcript_records": ("transcriptml.data.genomics", "extract_transcript_records"),
    "iter_gtf_records": ("transcriptml.data.genomics", "iter_gtf_records"),
    "load_transcript_features": ("transcriptml.data.genomics", "load_transcript_features"),
    "parse_gtf_attributes": ("transcriptml.data.genomics", "parse_gtf_attributes"),
    "RNA4": ("transcriptml.data.schemas", "RNA4"),
    "SALUKI6": ("transcriptml.data.schemas", "SALUKI6"),
    "SequenceSchema": ("transcriptml.data.schemas", "SequenceSchema"),
    "get_schema": ("transcriptml.data.schemas", "get_schema"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import a requested data symbol without importing unrelated Torch code."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'transcriptml.data' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
