from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _row_dict(row: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping-like row to a plain dictionary."""

    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def save_table(path: str | Path, rows: Sequence[Any]) -> None:
    """Save a sequence of dataclass or mapping rows as CSV."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [_row_dict(row) for row in rows]
    if not row_dicts:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        for row in row_dicts:
            writer.writerow(row)


def save_result_dir(
    out_dir: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    tables: Mapping[str, Sequence[Any]] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> None:
    """Save arrays, optional tables, and optional summary into a result directory."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", np.asarray(arr))
    for name, rows in (tables or {}).items():
        save_table(out / f"{name}.csv", rows)
    if summary is not None:
        (out / "summary.json").write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
