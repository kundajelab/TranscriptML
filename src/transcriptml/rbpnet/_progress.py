"""Progress helpers built on TranscriptML's native reporter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from transcriptml.progress import ProgressReporter, log_progress

T = TypeVar("T")


def track(
    iterable: Iterable[T],
    label: str,
    *,
    total: int | None = None,
    unit: str = "items",
    enabled: bool = True,
) -> Iterator[T]:
    """Yield an iterable while emitting throttled TranscriptML progress."""

    reporter = ProgressReporter(label, total=total, unit=unit, enabled=enabled)
    try:
        for value in iterable:
            yield value
            reporter.update()
    finally:
        reporter.close()


__all__ = ["ProgressReporter", "log_progress", "track"]
