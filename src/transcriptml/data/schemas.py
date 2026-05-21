from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class SequenceSchema:
    """Channel schema for an encoded RNA tensor.

    Arrays in TranscriptML use PyTorch convention: ``(N, C, L)`` for batches and
    ``(C, L)`` for a single sequence.
    """

    name: str
    channels: Tuple[str, ...]
    base_channels: Tuple[str, ...] = ("A", "C", "G", "U")
    description: str = ""

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_base_channels(self) -> int:
        return len(self.base_channels)

    @property
    def annotation_channels(self) -> Tuple[str, ...]:
        return self.channels[self.n_base_channels :]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SequenceSchema":
        return cls(
            name=str(data["name"]),
            channels=tuple(str(x) for x in data["channels"]),
            base_channels=tuple(str(x) for x in data.get("base_channels", ("A", "C", "G", "U"))),
            description=str(data.get("description", "")),
        )


RNA4 = SequenceSchema(
    name="rna4",
    channels=("A", "C", "G", "U"),
    base_channels=("A", "C", "G", "U"),
    description="RNA one-hot channels; T is treated as U and unknown bases are all-zero.",
)

SALUKI6 = SequenceSchema(
    name="saluki6",
    channels=("A", "C", "G", "U", "CDS_codon_start", "splice_junction"),
    base_channels=("A", "C", "G", "U"),
    description="RNA4 plus codon-start/CDS and splice-junction annotation channels.",
)

SCHEMAS: Dict[str, SequenceSchema] = {s.name: s for s in (RNA4, SALUKI6)}


def get_schema(schema: str | SequenceSchema) -> SequenceSchema:
    if isinstance(schema, SequenceSchema):
        return schema
    try:
        return SCHEMAS[schema]
    except KeyError as exc:
        raise ValueError(f"Unknown schema '{schema}'. Available: {sorted(SCHEMAS)}") from exc


def list_schemas() -> Sequence[str]:
    return tuple(sorted(SCHEMAS))
