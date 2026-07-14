"""TranscriptML: RNA sequence-to-function modeling utilities."""

__all__ = ["SequenceSchema", "get_schema"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily expose common top-level symbols without slowing CLI startup."""

    if name in {"SequenceSchema", "get_schema"}:
        from transcriptml.data.schemas import SequenceSchema, get_schema

        values = {"SequenceSchema": SequenceSchema, "get_schema": get_schema}
        return values[name]
    raise AttributeError(f"module 'transcriptml' has no attribute {name!r}")
