from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from transcriptml.data.schemas import SequenceSchema, get_schema


@dataclass
class DatasetBundle:
    """Self-describing processed dataset."""

    X: np.ndarray
    y: np.ndarray | None = None
    ids: Sequence[str] | None = None
    schema: SequenceSchema | str = "rna4"
    metadata: Sequence[Mapping[str, Any]] | None = None
    splits: Mapping[str, Sequence[int]] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize schema and validate array-aligned fields."""

        self.schema = get_schema(self.schema)
        if self.ids is None:
            self.ids = [str(i) for i in range(int(self.X.shape[0]))]
        if len(self.ids) != int(self.X.shape[0]):
            raise ValueError("ids length must match X.shape[0]")
        if self.y is not None and int(self.y.shape[0]) != int(self.X.shape[0]):
            raise ValueError("y length must match X.shape[0]")


def _json_default(obj: Any) -> Any:
    """Convert common NumPy and path objects to JSON-serializable values."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_bundle_metadata(bundle: DatasetBundle, out_dir: str | Path) -> None:
    """Write dataset sidecar metadata files for an existing ``X.npy``."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.txt").write_text("\n".join(str(x) for x in bundle.ids) + "\n", encoding="utf-8")
    if bundle.metadata is not None:
        (out / "metadata.json").write_text(
            json.dumps(list(bundle.metadata), indent=2, default=_json_default),
            encoding="utf-8",
        )
    if bundle.splits is not None:
        (out / "splits.json").write_text(
            json.dumps({k: [int(i) for i in v] for k, v in bundle.splits.items()}, indent=2),
            encoding="utf-8",
        )
    (out / "schema.json").write_text(json.dumps(bundle.schema.to_dict(), indent=2), encoding="utf-8")
    config = dict(bundle.config)
    config.setdefault("n_examples", int(bundle.X.shape[0]))
    config.setdefault("shape", [int(x) for x in bundle.X.shape])
    (out / "config.json").write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")


def save_bundle(bundle: DatasetBundle, out_dir: str | Path) -> None:
    """Write a complete dataset bundle, including arrays and sidecars."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", bundle.X)
    if bundle.y is not None:
        np.save(out / "y.npy", bundle.y)
    save_bundle_metadata(bundle, out)


def load_bundle(path: str | Path, *, mmap_mode: str | None = None) -> DatasetBundle:
    """Load a processed dataset bundle from disk."""

    root = Path(path)
    X = np.load(root / "X.npy", mmap_mode=mmap_mode)
    y_path = root / "y.npy"
    y = np.load(y_path, mmap_mode=mmap_mode) if y_path.exists() else None
    ids = (root / "ids.txt").read_text(encoding="utf-8").splitlines()
    schema = SequenceSchema.from_dict(json.loads((root / "schema.json").read_text(encoding="utf-8")))
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    splits_path = root / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else None
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return DatasetBundle(X=X, y=y, ids=ids, schema=schema, metadata=metadata, splits=splits, config=config)
