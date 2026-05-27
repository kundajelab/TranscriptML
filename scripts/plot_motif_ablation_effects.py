#!/usr/bin/env python3
"""Plot per-fold motif ablation effect distributions.

This script expects the TranscriptML motif-ablation output layout:

    motif_ablation/
      ARE/fold0/effects.npy
      ARE/fold1/effects.npy
      PRE/fold0/effects.npy
      ...

It writes one box-with-points plot per fold.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [part.strip() for part in value.split(",") if part.strip()]
    return out or None


def _parse_label_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    out: dict[str, str] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Label map entries must look like motif=label, got: {part!r}")
        key, label = part.split("=", 1)
        out[key.strip()] = label.strip()
    return out


def _prettify_label(label: str, label_map: dict[str, str]) -> str:
    if label in label_map:
        return label_map[label]
    out = label
    out = re.sub(r"_7mer_m8$", "", out)
    out = re.sub(r"_nonamer$", "", out)
    out = re.sub(r"^random_", "", out)
    out = out.replace("miR", "mir")
    return out


def _discover_motifs(root: Path) -> list[str]:
    motifs = []
    for path in root.iterdir():
        if path.is_dir() and any(path.glob("fold*/effects.npy")):
            motifs.append(path.name)
    return sorted(motifs, key=_natural_key)


def _discover_folds(root: Path, motifs: Iterable[str]) -> list[str]:
    folds = set()
    for motif in motifs:
        for effects_path in (root / motif).glob("fold*/effects.npy"):
            folds.add(effects_path.parent.name)
    return sorted(folds, key=_natural_key)


def _load_effects(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _fold_data(root: Path, motifs: list[str], fold: str) -> list[np.ndarray]:
    data = []
    for motif in motifs:
        path = root / motif / fold / "effects.npy"
        if path.exists():
            data.append(_load_effects(path))
        else:
            print(f"[plot_motif_ablation_effects] missing {path}", file=sys.stderr)
            data.append(np.array([], dtype=float))
    return data


def _auto_ylim(data: list[np.ndarray]) -> tuple[float, float]:
    values = [arr for arr in data if arr.size]
    if not values:
        return -1.0, 1.0
    all_values = np.concatenate(values)
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = max(hi - lo, 0.05)
    pad = max(span * 0.14, 0.02)
    return lo - pad, hi + pad


def _plot_fold(
    *,
    data: list[np.ndarray],
    motifs: list[str],
    fold: str,
    out_path: Path,
    label_map: dict[str, str],
    title_template: str | None,
    ylabel: str,
    xlabel: str,
    ylim: tuple[float, float] | None,
    dpi: int,
    max_points_per_motif: int | None,
    seed: int,
) -> None:
    labels = [_prettify_label(motif, label_map) for motif in motifs]
    n_motifs = len(motifs)
    fig_width = max(8.0, 0.85 * n_motifs + 2.0)
    fig_height = 6.8
    rng = np.random.default_rng(seed + sum(ord(c) for c in fold))

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    positions = np.arange(1, n_motifs + 1)
    nonempty = [(pos, arr) for pos, arr in zip(positions, data) if arr.size]
    if nonempty:
        ax.boxplot(
            [arr for _, arr in nonempty],
            positions=[pos for pos, _ in nonempty],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#bfbfbf", "edgecolor": "black", "linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 2.2},
            whiskerprops={"color": "black", "linewidth": 1.2},
            capprops={"color": "black", "linewidth": 1.2},
        )

    for pos, arr in zip(positions, data):
        if not arr.size:
            continue
        points = arr
        if max_points_per_motif is not None and arr.size > max_points_per_motif:
            idx = rng.choice(arr.size, size=int(max_points_per_motif), replace=False)
            points = arr[idx]
        jitter = rng.uniform(-0.30, 0.30, size=points.size)
        ax.scatter(
            np.full(points.size, pos) + jitter,
            points,
            s=5,
            alpha=0.35,
            color="#0b7894",
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    ax.axhline(0.0, color="#b00000", linewidth=1.4, linestyle=(0, (1.0, 3.0)), zorder=0)
    ax.set_xlim(0.4, n_motifs + 0.6)
    if ylim is None:
        ylim = _auto_ylim(data)
    ax.set_ylim(*ylim)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    count_y = y_max - 0.06 * y_span
    for pos, arr in zip(positions, data):
        ax.text(pos, count_y, f"n = {arr.size}", ha="center", va="top", fontsize=11)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title_template:
        ax.set_title(title_template.format(fold=fold), pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.5, length=4)
    ax.grid(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motif_ablation_root",
        help="Directory containing motif subdirectories, e.g. ${INTERPRET_ROOT}/motif_ablation",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots. Defaults to <motif_ablation_root>/plots.",
    )
    parser.add_argument(
        "--motif-order",
        help="Comma-separated motif directory names. Defaults to all discovered motifs sorted naturally.",
    )
    parser.add_argument(
        "--folds",
        help="Comma-separated fold directory names, e.g. fold0,fold1. Defaults to all discovered folds.",
    )
    parser.add_argument(
        "--label-map",
        help="Comma-separated label overrides, e.g. ARE_nonamer=ARE,random_ctl1=ctl1.",
    )
    parser.add_argument("--prefix", default="motif_ablation_effects", help="Output filename prefix.")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output figure format.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"))
    parser.add_argument(
        "--max-points-per-motif",
        type=int,
        help="Randomly downsample plotted points per motif. Boxes and n labels still use all effects.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for point jitter and optional downsampling.")
    parser.add_argument("--title-template", default="", help="Optional title, may include {fold}.")
    parser.add_argument("--xlabel", default="motif")
    parser.add_argument("--ylabel", default="Effect of ablating motif instance")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.motif_ablation_root)
    if not root.exists():
        raise SystemExit(f"Motif ablation root does not exist: {root}")

    motif_order = _parse_csv(args.motif_order) or _discover_motifs(root)
    if not motif_order:
        raise SystemExit(f"No motif ablation outputs found under {root}")

    missing_motifs = [motif for motif in motif_order if not (root / motif).is_dir()]
    if missing_motifs:
        raise SystemExit(f"Motif directories not found under {root}: {', '.join(missing_motifs)}")

    folds = _parse_csv(args.folds) or _discover_folds(root, motif_order)
    if not folds:
        raise SystemExit(f"No fold*/effects.npy files found under {root}")

    out_dir = Path(args.out_dir) if args.out_dir else root / "plots"
    label_map = _parse_label_map(args.label_map)
    ylim = tuple(args.ylim) if args.ylim else None

    for fold in folds:
        data = _fold_data(root, motif_order, fold)
        out_path = out_dir / f"{args.prefix}_{fold}.{args.format}"
        _plot_fold(
            data=data,
            motifs=motif_order,
            fold=fold,
            out_path=out_path,
            label_map=label_map,
            title_template=args.title_template or None,
            ylabel=args.ylabel,
            xlabel=args.xlabel,
            ylim=ylim,
            dpi=args.dpi,
            max_points_per_motif=args.max_points_per_motif,
            seed=args.seed,
        )
        print(f"[plot_motif_ablation_effects] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
