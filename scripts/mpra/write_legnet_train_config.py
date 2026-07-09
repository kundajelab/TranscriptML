#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a TranscriptML LegNet train config for one MPRA dataset.")
    parser.add_argument("--dataset", required=True, help="TranscriptML MPRA dataset bundle")
    parser.add_argument("--base-config", required=True, help="Base TranscriptML train config JSON")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--config-path", required=True, help="Generated train config path")
    parser.add_argument("--seed", type=int, help="Optional seed override")
    args = parser.parse_args()

    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    config["dataset"] = str(Path(args.dataset))
    config["output_dir"] = str(Path(args.output_dir))
    config.setdefault("model", {"name": "legnet", "params": {}})
    config.setdefault("mmap_mode", "r")
    if args.seed is not None:
        config["seed"] = int(args.seed)

    config_path = Path(args.config_path)
    _write_json(config_path, config)
    print(config_path)


if __name__ == "__main__":
    main()
