#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


TOP_LEVEL_CONFIG_KEYS = {
    "dataset",
    "output_dir",
    "model",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "patience",
    "monitor",
    "loss",
    "device",
    "num_workers",
    "mmap_mode",
    "seed",
    "progress",
    "sequence_controls",
    "split",
}

METADATA_COLUMNS = {"combo_id", "combo_label", "label", "notes", "description"}

GRID_PRESETS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "saluki": {
        "smoke": {
            "filters": [32, 64],
            "learning_rate": [1e-3],
            "epochs": [3],
            "patience": [2],
        },
        "standard": {
            "filters": [64],
            "dropout": [0.2, 0.3],
            "learning_rate": [1e-3, 3e-4],
            "weight_decay": [0.0, 1e-2],
            "batch_size": [64],
            "patience": [8],
        },
    },
    "legnet": {
        "smoke": {
            "stem_ch": [32, 64],
            "learning_rate": [1e-3],
            "epochs": [3],
            "patience": [2],
        },
        "standard": {
            "stem_ch": [64],
            "block_dropout": [0.0, 0.1],
            "head_dropout": [0.1, 0.2],
            "learning_rate": [1e-3, 3e-4],
            "weight_decay": [0.0, 1e-2],
            "batch_size": [64],
            "patience": [8],
        },
    },
}


def parse_cell_value(text: str) -> Any:
    stripped = str(text).strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def format_cell_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False
    opens = {"[", "{", "("}
    closes = {"]", "}", ")"}

    for char in text:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\" and quote:
            current.append(char)
            escape = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char in opens:
            depth += 1
        elif char in closes and depth > 0:
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    parts.append("".join(current).strip())
    return parts


def parse_param_spec(spec: str) -> tuple[str, list[Any]]:
    name, sep, values_text = spec.partition("=")
    name = name.strip()
    if not sep or not name:
        raise ValueError(f"Expected --param NAME=VALUE[,VALUE...], got: {spec}")
    values = [parse_cell_value(part) for part in split_top_level(values_text) if part.strip()]
    if not values:
        raise ValueError(f"--param {name} must include at least one value")
    return name, values


def detect_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".tab"}:
        return "\t"
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_table(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    table_path = Path(path)
    delimiter = detect_delimiter(table_path)
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Sweep table has no header: {table_path}")
        headers = [str(name).strip() for name in reader.fieldnames]
        if any(not name for name in headers):
            raise ValueError(f"Sweep table has an empty column name: {table_path}")
        rows: list[dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                extra = row[None]
                raise ValueError(f"Row {line_no} has extra fields not present in the header: {extra}")
            clean = {header: (row.get(header) or "").strip() for header in headers}
            if any(value != "" for value in clean.values()):
                rows.append(clean)
    return headers, rows


def write_table(path: str | Path, headers: Iterable[str], rows: Iterable[Mapping[str, Any]], *, delimiter: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header_list = list(headers)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_list, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header_list})
    return out


def normalize_table(table: str | Path, out: str | Path) -> Path:
    headers, rows = read_table(table)
    return write_table(out, headers, rows, delimiter="\t")


def generate_grid(model: str, preset: str, params: Iterable[str], out: str | Path) -> Path:
    if model not in GRID_PRESETS:
        raise ValueError(f"Unknown model grid '{model}'. Available: {sorted(GRID_PRESETS)}")
    if preset not in GRID_PRESETS[model]:
        raise ValueError(f"Unknown preset '{preset}' for {model}. Available: {sorted(GRID_PRESETS[model])}")
    grid = {name: list(values) for name, values in GRID_PRESETS[model][preset].items()}
    for spec in params:
        name, values = parse_param_spec(spec)
        grid[name] = values

    headers = list(grid)
    rows = []
    for combo in itertools.product(*(grid[name] for name in headers)):
        rows.append({name: format_cell_value(value) for name, value in zip(headers, combo)})

    delimiter = "," if Path(out).suffix.lower() == ".csv" else "\t"
    return write_table(out, headers, rows, delimiter=delimiter)


def _ensure_model_mapping(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if isinstance(model, dict):
        model.setdefault("params", {})
        if model["params"] is None:
            model["params"] = {}
        return model
    if model is None:
        model = {"params": {}}
    else:
        model = {"name": str(model), "params": {}}
    config["model"] = model
    return model


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = [part.strip() for part in key.split(".") if part.strip()]
    if not parts:
        return
    if parts[0] == "model":
        _ensure_model_mapping(config)
    current: dict[str, Any] = config
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def apply_hparams(config: dict[str, Any], row: Mapping[str, str]) -> dict[str, Any]:
    for raw_key, raw_value in row.items():
        key = str(raw_key).strip()
        value_text = "" if raw_value is None else str(raw_value).strip()
        if not key or value_text == "" or key.startswith("_") or key in METADATA_COLUMNS:
            continue
        value = parse_cell_value(value_text)
        if "." in key:
            _set_dotted(config, key, value)
        elif key == "model" and not isinstance(value, dict):
            model = _ensure_model_mapping(config)
            model["name"] = str(value)
        elif key in TOP_LEVEL_CONFIG_KEYS:
            config[key] = value
        else:
            model = _ensure_model_mapping(config)
            params = model.setdefault("params", {})
            if not isinstance(params, dict):
                params = {}
                model["params"] = params
            params[key] = value
    return config


def parsed_hparams(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        str(key): parse_cell_value(value)
        for key, value in row.items()
        if str(key).strip() and value is not None and str(value).strip() != ""
    }


def _replace_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def combo_name(index: int, width: int = 4) -> str:
    return f"combo_{int(index):0{int(width)}d}"


def write_fold_artifacts(
    *,
    dataset: str | Path,
    base_config: str | Path,
    cv_root: str | Path,
    fold: int,
    n_folds: int,
    seed: int,
    val_offset: int = 1,
    sweep_table: str | Path | None = None,
    combo_index: int | None = None,
    combo_id_width: int = 4,
    default_model_name: str | None = None,
) -> Path:
    import numpy as np

    dataset_path = Path(dataset)
    cv_root_path = Path(cv_root)

    if int(n_folds) <= 1:
        raise ValueError("n_folds must be greater than 1")
    if int(fold) < 0 or int(fold) >= int(n_folds):
        raise ValueError(f"fold must be in [0, {int(n_folds) - 1}], got {fold}")

    row: dict[str, str] | None = None
    combo_dir: Path | None = None
    if sweep_table is not None:
        if combo_index is None:
            raise ValueError("combo_index is required when sweep_table is provided")
        _, rows = read_table(sweep_table)
        if int(combo_index) < 0 or int(combo_index) >= len(rows):
            raise ValueError(f"combo_index {combo_index} is out of range for {len(rows)} sweep rows")
        row = rows[int(combo_index)]
        combo_dir = cv_root_path / combo_name(int(combo_index), combo_id_width)
        fold_dir = combo_dir / f"fold{int(fold)}"
        _write_json(combo_dir / "hparams.json", parsed_hparams(row))
    else:
        fold_dir = cv_root_path / f"fold{int(fold)}"

    fold_dataset = fold_dir / "dataset"
    model_dir = fold_dir / "model"
    fold_dataset.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name in ("X.npy", "y.npy", "ids.txt", "schema.json", "metadata.json", "config.json"):
        src = dataset_path / name
        if src.exists():
            _replace_link(src, fold_dataset / name)

    n_examples = int(np.load(dataset_path / "X.npy", mmap_mode="r").shape[0])
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n_examples, dtype=np.int64)
    rng.shuffle(indices)
    folds = [fold_indices.astype(int).tolist() for fold_indices in np.array_split(indices, int(n_folds))]

    test_fold = int(fold)
    val_fold = (test_fold + int(val_offset)) % int(n_folds)
    train: list[int] = []
    for i, fold_indices in enumerate(folds):
        if i not in {test_fold, val_fold}:
            train.extend(fold_indices)

    splits = {"train": train, "val": folds[val_fold], "test": folds[test_fold]}
    _write_json(fold_dataset / "splits.json", splits)

    config = json.loads(Path(base_config).read_text(encoding="utf-8"))
    if default_model_name is not None:
        config.setdefault("model", {"name": default_model_name, "params": {}})
    if row is not None:
        apply_hparams(config, row)
    config["dataset"] = str(fold_dataset)
    config["output_dir"] = str(model_dir)
    config["seed"] = int(config.get("seed", seed)) + int(fold)
    config.setdefault("mmap_mode", "r")

    config_path = fold_dir / "train_config.json"
    _write_json(config_path, config)
    return config_path


def _parse_combo_index(path: Path) -> int | None:
    match = re.fullmatch(r"combo_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _fold_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"fold(\d+)", path.name)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _format_summary_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def _collect_combo_summary(combo_dir: Path, combo_index: int, hparams: Mapping[str, Any]) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    metric_values = {"test_pearson": [], "test_mse": [], "test_loss": []}
    if combo_dir.exists():
        for fold_dir in sorted((path for path in combo_dir.glob("fold*") if path.is_dir()), key=_fold_sort_key):
            summary_path = fold_dir / "model" / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            fold_match = re.fullmatch(r"fold(\d+)", fold_dir.name)
            fold_row = {
                "fold": int(fold_match.group(1)) if fold_match else fold_dir.name,
                "summary_path": str(summary_path),
            }
            for metric in metric_values:
                value = _finite_float(summary.get(metric))
                fold_row[metric] = value
                if value is not None:
                    metric_values[metric].append(value)
            fold_rows.append(fold_row)

    aggregate: dict[str, Any] = {}
    for metric, values in metric_values.items():
        mean, std = _mean_std(values)
        aggregate[f"mean_{metric}"] = mean
        aggregate[f"std_{metric}"] = std

    return {
        "combo_index": combo_index,
        "combo_dir": str(combo_dir),
        "hparams": dict(hparams),
        "completed_folds": len(fold_rows),
        "folds": fold_rows,
        **aggregate,
    }


def summarize_sweep(
    *,
    sweep_root: str | Path,
    sweep_table: str | Path | None = None,
    out: str | Path | None = None,
    combo_id_width: int = 4,
) -> Path:
    root = Path(sweep_root)
    table_path = Path(sweep_table) if sweep_table is not None else root / "sweep_table.tsv"
    table_headers: list[str] = []
    table_rows: list[dict[str, str]] = []
    if table_path.exists():
        table_headers, table_rows = read_table(table_path)

    combo_indices = set(range(len(table_rows)))
    if root.exists():
        for combo_dir in root.glob("combo_*"):
            index = _parse_combo_index(combo_dir)
            if index is not None:
                combo_indices.add(index)

    records: list[dict[str, Any]] = []
    for index in sorted(combo_indices):
        combo_dir = root / combo_name(index, combo_id_width)
        raw_hparams: Mapping[str, Any]
        if index < len(table_rows):
            raw_hparams = table_rows[index]
        else:
            hparams_path = combo_dir / "hparams.json"
            raw_hparams = json.loads(hparams_path.read_text(encoding="utf-8")) if hparams_path.exists() else {}
        summary = _collect_combo_summary(combo_dir, index, raw_hparams)
        if combo_dir.exists():
            _write_json(combo_dir / "combo_summary.json", summary)
        records.append(summary)

    def rank_key(record: Mapping[str, Any]) -> tuple[int, float, float, int]:
        pearson = _finite_float(record.get("mean_test_pearson"))
        mse = _finite_float(record.get("mean_test_mse"))
        return (
            0 if pearson is not None else 1,
            -pearson if pearson is not None else math.inf,
            mse if mse is not None else math.inf,
            int(record["combo_index"]),
        )

    ranked = sorted(records, key=rank_key)
    ranks = {id(record): rank + 1 for rank, record in enumerate(ranked)}

    param_headers = list(table_headers)
    if not param_headers:
        seen: set[str] = set()
        for record in records:
            for key in record.get("hparams", {}):
                if key not in seen:
                    seen.add(key)
                    param_headers.append(str(key))

    fieldnames = [
        "rank",
        "combo_index",
        "combo_dir",
        "completed_folds",
        "mean_test_pearson",
        "std_test_pearson",
        "mean_test_mse",
        "std_test_mse",
        "mean_test_loss",
        "std_test_loss",
        *param_headers,
    ]

    rows = []
    for record in records:
        hparams = record.get("hparams", {})
        row = {
            "rank": ranks[id(record)],
            "combo_index": record["combo_index"],
            "combo_dir": record["combo_dir"],
            "completed_folds": record["completed_folds"],
            "mean_test_pearson": _format_summary_value(record.get("mean_test_pearson")),
            "std_test_pearson": _format_summary_value(record.get("std_test_pearson")),
            "mean_test_mse": _format_summary_value(record.get("mean_test_mse")),
            "std_test_mse": _format_summary_value(record.get("std_test_mse")),
            "mean_test_loss": _format_summary_value(record.get("mean_test_loss")),
            "std_test_loss": _format_summary_value(record.get("std_test_loss")),
        }
        for name in param_headers:
            value = hparams.get(name, "")
            row[name] = value if isinstance(value, str) else format_cell_value(value)
        rows.append(row)

    out_path = Path(out) if out is not None else root / "sweep_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    write_table(tmp_path, fieldnames, rows, delimiter="\t")
    tmp_path.replace(out_path)
    return out_path


def count_rows_cli(args: argparse.Namespace) -> None:
    _, rows = read_table(args.table)
    print(len(rows))


def normalize_table_cli(args: argparse.Namespace) -> None:
    print(normalize_table(args.table, args.out))


def generate_grid_cli(model: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Write a {model} hyperparameter sweep table.")
    parser.add_argument("--preset", choices=sorted(GRID_PRESETS[model]), default="standard")
    parser.add_argument("--param", action="append", default=[], help="Override/add a grid dimension: NAME=VALUE[,VALUE...]")
    parser.add_argument("--out", required=True, help="Output CSV/TSV path; suffix controls delimiter")
    args = parser.parse_args(argv)
    print(generate_grid(model, args.preset, args.param, args.out))


def write_fold_artifacts_cli(args: argparse.Namespace) -> None:
    config_path = write_fold_artifacts(
        dataset=args.dataset,
        base_config=args.base_config,
        cv_root=args.cv_root,
        fold=args.fold,
        n_folds=args.n_folds,
        seed=args.seed,
        val_offset=args.val_offset,
        sweep_table=args.sweep_table,
        combo_index=args.combo_index,
        combo_id_width=args.combo_id_width,
        default_model_name=args.default_model_name,
    )
    print(config_path)


def summarize_sweep_cli(args: argparse.Namespace) -> None:
    print(
        summarize_sweep(
            sweep_root=args.sweep_root,
            sweep_table=args.sweep_table,
            out=args.out,
            combo_id_width=args.combo_id_width,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for TranscriptML script-level hyperparameter sweeps.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("count-rows", help="Count non-empty data rows in a CSV/TSV sweep table")
    p.add_argument("--table", required=True)
    p.set_defaults(func=count_rows_cli)

    p = sub.add_parser("normalize-table", help="Rewrite a CSV/TSV sweep table as normalized TSV")
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=normalize_table_cli)

    p = sub.add_parser("generate-grid", help="Write a built-in hyperparameter grid")
    p.add_argument("--model", choices=sorted(GRID_PRESETS), required=True)
    p.add_argument("--preset", default="standard")
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--out", required=True)
    p.set_defaults(func=lambda args: print(generate_grid(args.model, args.preset, args.param, args.out)))

    p = sub.add_parser("write-fold-artifacts", help="Write one fold dataset bundle and train config")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--cv-root", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-offset", type=int, default=1)
    p.add_argument("--sweep-table")
    p.add_argument("--combo-index", type=int)
    p.add_argument("--combo-id-width", type=int, default=4)
    p.add_argument("--default-model-name")
    p.set_defaults(func=write_fold_artifacts_cli)

    p = sub.add_parser("summarize-sweep", help="Aggregate combo/fold model summaries into sweep_summary.tsv")
    p.add_argument("--sweep-root", required=True)
    p.add_argument("--sweep-table")
    p.add_argument("--out")
    p.add_argument("--combo-id-width", type=int, default=4)
    p.set_defaults(func=summarize_sweep_cli)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
