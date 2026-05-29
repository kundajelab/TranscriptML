from __future__ import annotations

import sys
import time
from typing import TextIO


def log_progress(message: str, *, enabled: bool = True, stream: TextIO | None = None) -> None:
    """Print a simple TranscriptML progress message.

    Args:
        message: Human-readable progress text to emit.
        enabled: Whether to print the message.
        stream: Optional text stream to write to. Defaults to ``sys.stderr``.
    """

    if not enabled:
        return
    out = sys.stderr if stream is None else stream
    print(f"[transcriptml] {message}", file=out, flush=True)


class ProgressReporter:
    """Small stderr progress reporter without an external dependency."""

    def __init__(
        self,
        label: str,
        *,
        total: int | None = None,
        unit: str = "items",
        enabled: bool = True,
        min_interval: float = 30.0,
        percent_step: float = 5.0,
        stream: TextIO | None = None,
    ):
        """Create a throttled progress reporter.

        Args:
            label: Prefix label shown in each progress line.
            total: Optional total item count used for percentage reporting.
            unit: Human-readable unit name for counted items.
            enabled: Whether to emit progress lines.
            min_interval: Minimum seconds between non-forced progress lines.
            percent_step: Percentage increment that triggers progress emission
                when ``total`` is known.
            stream: Optional text stream to write to. Defaults to ``sys.stderr``.
        """

        self.label = label
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.enabled = bool(enabled)
        self.min_interval = float(min_interval)
        self.percent_step = float(percent_step)
        self.stream = sys.stderr if stream is None else stream
        self.current = 0
        self._last_emit = 0.0
        self._next_percent = self.percent_step
        self._closed = False
        if self.enabled:
            if self.total is None:
                self._emit("started")
            else:
                self._emit(f"started (0/{self.total} {self.unit})")

    def _emit(self, text: str) -> None:
        """Emit one formatted progress line.

        Args:
            text: Progress status text appended after the reporter label.
        """

        self._last_emit = time.monotonic()
        print(f"[transcriptml] {self.label}: {text}", file=self.stream, flush=True)

    def update(
        self,
        advance: int = 1,
        *,
        current: int | None = None,
        extra: str | None = None,
        force: bool = False,
    ) -> None:
        """Advance progress and emit a line when enough work has accumulated.

        Args:
            advance: Amount to add to the current count when ``current`` is not
                supplied.
            current: Optional absolute current count to set.
            extra: Optional text appended to the emitted progress line.
            force: Whether to emit a line regardless of throttling thresholds.
        """

        if current is None:
            self.current += int(advance)
        else:
            self.current = int(current)
        if not self.enabled or self._closed:
            return

        now = time.monotonic()
        should_emit = bool(force)
        if self.total is None:
            should_emit = should_emit or (now - self._last_emit >= self.min_interval)
            text = f"{self.current} {self.unit}"
        else:
            total = max(1, self.total)
            percent = min(100.0, 100.0 * self.current / total)
            should_emit = (
                should_emit
                or self.current >= self.total
                or percent >= self._next_percent
                or now - self._last_emit >= self.min_interval
            )
            text = f"{min(self.current, self.total)}/{self.total} {self.unit} ({percent:.1f}%)"
            while percent >= self._next_percent:
                self._next_percent += self.percent_step
        if extra:
            text = f"{text}; {extra}"
        if should_emit:
            self._emit(text)

    def close(self, *, extra: str | None = None) -> None:
        """Emit a final progress line.

        Args:
            extra: Optional text appended to the final progress line.
        """

        if not self.enabled or self._closed:
            return
        if self.total is None:
            text = f"finished ({self.current} {self.unit})"
        else:
            text = f"finished ({min(self.current, self.total)}/{self.total} {self.unit})"
        if extra:
            text = f"{text}; {extra}"
        self._emit(text)
        self._closed = True
