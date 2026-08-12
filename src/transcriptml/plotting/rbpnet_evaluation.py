"""Robust, deterministic plots for structured RBPNet evaluation reports."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-rbpnet-evaluation"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)

import matplotlib

if (
    "matplotlib.pyplot" not in sys.modules
    and "MPLBACKEND" not in os.environ
    and not os.environ.get("DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from transcriptml.rbpnet.dataset import RBPNetDataset, collate_rbpnet
from transcriptml.rbpnet.evaluation_metrics import safe_correlations


_TRACK_LABELS = {"pooled_ip": "Pooled IP", "sminput": "SMInput"}
_METRIC_LABELS = {
    "kl_per_read": "KL / read (nats)",
    "jsd": "JSD (nats)",
    "wasserstein_nt": "Wasserstein (nt)",
    "information_gain_uniform_per_read": "Information gain / read (nats)",
}


def _values(rows: Sequence[Mapping[str, object]], column: str) -> np.ndarray:
    return np.asarray([row.get(column, np.nan) for row in rows], dtype=np.float64)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _scatter_or_hexbin(ax, x: np.ndarray, y: np.ndarray, *, seed: int = 123) -> None:
    x, y = _finite_xy(np.asarray(x), np.asarray(y))
    if x.size == 0:
        ax.text(0.5, 0.5, "No informative observations", ha="center", va="center")
        return
    if x.size > 2_500:
        image = ax.hexbin(x, y, gridsize=45, mincnt=1, bins="log", cmap="viridis")
        ax.figure.colorbar(image, ax=ax, label="log10 bin count")
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(x.size)
        ax.scatter(x[order], y[order], s=10, alpha=0.35, edgecolors="none")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_profile_depth(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    metrics = ("kl_per_read", "jsd", "wasserstein_nt")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    for row_index, track in enumerate(("pooled_ip", "sminput")):
        depth = _values(rows, f"{track}_profile_count")
        x = np.log10(depth + 1.0)
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            _scatter_or_hexbin(ax, x, _values(rows, f"{track}_{metric}"))
            ax.set_xlabel("log10(observed profile reads + 1)")
            ax.set_ylabel(_METRIC_LABELS[metric])
            ax.set_title(f"{_TRACK_LABELS[track]}: {_METRIC_LABELS[metric]}")
    _save(fig, path)


def _plot_profile_distributions(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    metrics = (
        "kl_per_read",
        "jsd",
        "wasserstein_nt",
        "information_gain_uniform_per_read",
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for row_index, track in enumerate(("pooled_ip", "sminput")):
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            values = _values(rows, f"{track}_{metric}")
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=min(50, max(10, int(np.sqrt(values.size)))), alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No informative observations", ha="center", va="center")
            if track == "pooled_ip" and metric in {"jsd", "wasserstein_nt"}:
                ceiling = _values(rows, f"ip_replicate_ceiling_{metric}")
                ceiling = ceiling[np.isfinite(ceiling)]
                if ceiling.size:
                    ax.hist(
                        ceiling,
                        bins=min(50, max(10, int(np.sqrt(ceiling.size)))),
                        histtype="step",
                        linewidth=1.8,
                        label="replicate ceiling",
                    )
                    ax.legend(fontsize=8)
            ax.set_xlabel(_METRIC_LABELS[metric])
            ax.set_ylabel("Loci")
            ax.set_title(_TRACK_LABELS[track])
    _save(fig, path)


def _locus_calibration(
    calibration: Sequence[Mapping[str, object]], replicate: str
) -> list[Mapping[str, object]]:
    return [
        row
        for row in calibration
        if row.get("row_type") == "locus" and row.get("replicate") == replicate
    ]


def _bin_calibration(
    calibration: Sequence[Mapping[str, object]], replicate: str
) -> list[Mapping[str, object]]:
    return sorted(
        [
            row
            for row in calibration
            if row.get("row_type") == "bin" and row.get("replicate") == replicate
        ],
        key=lambda row: int(row.get("bin", 0)),
    )


def _plot_eta(
    calibration: Sequence[Mapping[str, object]],
    replicate_names: Sequence[str],
    path: Path,
) -> bool:
    panels = [name for name in replicate_names if _locus_calibration(calibration, name)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, replicate in zip(axes[0], panels):
        data = _locus_calibration(calibration, replicate)
        eta = np.asarray([row["eta"] for row in data], dtype=float)
        empirical = np.asarray([row["empirical_eta"] for row in data], dtype=float)
        _scatter_or_hexbin(ax, eta, empirical)
        correlations = safe_correlations(eta, empirical)
        ax.text(
            0.02,
            0.98,
            (
                f"Pearson={correlations['pearson']:.3g}\n"
                f"Spearman={correlations['spearman']:.3g}\n"
                f"n={correlations['n']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_xlabel("Predicted eta")
        ax.set_ylabel("Stabilized empirical eta")
        ax.set_title(str(replicate))
    _save(fig, path)
    return True


def _plot_calibration(
    calibration: Sequence[Mapping[str, object]],
    replicate_names: Sequence[str],
    path: Path,
) -> bool:
    panels = [name for name in replicate_names if _locus_calibration(calibration, name)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.8), squeeze=False)
    for ax, replicate in zip(axes[0], panels):
        data = _locus_calibration(calibration, replicate)
        predicted = np.asarray([row["predicted_probability"] for row in data], dtype=float)
        observed = np.asarray([row["observed_fraction"] for row in data], dtype=float)
        total = np.asarray([row["total_reads"] for row in data], dtype=float)
        keep = np.isfinite(predicted) & np.isfinite(observed) & (total > 0)
        if keep.sum() > 3_000:
            image = ax.hexbin(
                predicted[keep],
                observed[keep],
                C=np.log10(total[keep] + 1),
                reduce_C_function=np.mean,
                gridsize=42,
                mincnt=1,
                cmap="viridis",
            )
            fig.colorbar(image, ax=ax, label="mean log10(reads + 1)")
        else:
            image = ax.scatter(
                predicted[keep],
                observed[keep],
                c=np.log10(total[keep] + 1),
                s=14,
                alpha=0.45,
                cmap="viridis",
                edgecolors="none",
            )
            if keep.any():
                fig.colorbar(image, ax=ax, label="log10(reads + 1)")
        binned = _bin_calibration(calibration, replicate)
        if binned:
            ax.plot(
                [float(row["predicted_bin"]) for row in binned],
                [float(row["observed_bin"]) for row in binned],
                marker="o",
                color="#d62728",
                linewidth=2,
                label="read-weighted bins",
            )
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted IP fraction")
        ax.set_ylabel("Observed IP / (IP + SMInput)")
        ax.set_title(str(replicate))
        ax.legend(fontsize=8)
    _save(fig, path)
    return True


def _normalized(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[~mask] = np.nan
    total = np.nansum(result)
    if total > 0:
        result /= total
    return result


@torch.no_grad()
def _predict_example(
    dataset: RBPNetDataset,
    model: torch.nn.Module,
    index: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    item = dataset.item_for_shift(index, 0)
    batch = collate_rbpnet([item]).to(device)
    output = model(
        batch.sequence,
        measurement_mask=batch.measurement_mask,
        profile_mask=batch.profile_valid_mask,
    )
    predicted = {
        "target": output.target_probs[0].detach().cpu().numpy(),
        "control": output.control_probs[0].detach().cpu().numpy(),
        "ip": output.ip_probs[0].detach().cpu().numpy(),
    }
    return item, predicted


def _plot_representatives(
    representative: Sequence[Mapping[str, object]],
    dataset: RBPNetDataset,
    model: torch.nn.Module,
    device: torch.device,
    path: Path,
    saved_profile_paths: Mapping[str, Path] | None,
) -> bool:
    if not representative:
        return False
    saved = (
        {
            name: np.load(profile_path, mmap_mode="r", allow_pickle=False)
            for name, profile_path in saved_profile_paths.items()
        }
        if saved_profile_paths is not None
        else None
    )
    fig, axes = plt.subplots(
        len(representative), 2, figsize=(13, max(3.0, 2.6 * len(representative))), squeeze=False
    )
    for row_index, selected in enumerate(representative):
        original_index = int(selected["index"])
        item = dataset.item_for_shift(original_index, 0)
        if saved is None:
            _, predicted = _predict_example(dataset, model, original_index, device)
        else:
            evaluation_row = int(selected["evaluation_row"])
            predicted = {
                name: np.asarray(array[evaluation_row]) for name, array in saved.items()
            }
        mask = np.asarray(item["profile_valid_mask"], dtype=bool)
        x = np.arange(mask.size)
        ip_ax, sm_ax = axes[row_index]
        observed_ip = _normalized(np.asarray(item["pooled_ip_profile"]), mask)
        observed_sm = _normalized(np.asarray(item["sminput_profile"]), mask)
        ip_ax.plot(x, observed_ip, color="black", linewidth=1.2, label="observed pooled IP")
        ip_ax.plot(x, _normalized(predicted["ip"], mask), label="predicted IP")
        ip_ax.plot(
            x,
            _normalized(predicted["target"], mask),
            linestyle=":",
            label="latent target",
        )
        sm_ax.plot(x, observed_sm, color="black", linewidth=1.2, label="observed SMInput")
        sm_ax.plot(x, _normalized(predicted["control"], mask), label="predicted control")
        title = (
            f"{selected['tier']}: {selected['example_id']} | "
            f"IP KL/read={float(selected['metric_value']):.3g}"
        )
        ip_ax.set_title(title, fontsize=9)
        sm_ax.set_title("Control profile", fontsize=9)
        for ax in (ip_ax, sm_ax):
            ax.set_xlabel("Profile position (nt)")
            ax.set_ylabel("Normalized probability")
            ax.legend(fontsize=7, loc="upper right")
    _save(fig, path)
    return True


def _plot_stratified(rows: Sequence[Mapping[str, object]], path: Path) -> bool:
    fields = [
        field
        for field in ("selection_state", "region_type")
        if any(row.get(field) not in {None, ""} for row in rows)
    ]
    if not fields:
        return False
    metrics = ("kl_per_read", "jsd", "wasserstein_nt")
    fig, axes = plt.subplots(
        len(fields), 3, figsize=(16, 4.2 * len(fields)), squeeze=False
    )
    for row_index, field in enumerate(fields):
        categories = sorted({str(row[field]) for row in rows if row.get(field) not in {None, ""}})
        positions = np.arange(len(categories), dtype=float)
        for column_index, metric in enumerate(metrics):
            ax = axes[row_index, column_index]
            for track_index, track in enumerate(("pooled_ip", "sminput")):
                means = []
                counts = []
                for category in categories:
                    values = np.asarray(
                        [
                            row.get(f"{track}_{metric}", np.nan)
                            for row in rows
                            if str(row.get(field, "")) == category
                        ],
                        dtype=float,
                    )
                    values = values[np.isfinite(values)]
                    means.append(float(values.mean()) if values.size else np.nan)
                    counts.append(int(values.size))
                offset = (-0.19, 0.19)[track_index]
                bars = ax.bar(
                    positions + offset,
                    means,
                    width=0.36,
                    label=_TRACK_LABELS[track],
                )
                for bar, count in zip(bars, counts):
                    if np.isfinite(bar.get_height()):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{count}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )
            ax.set_xticks(positions, categories, rotation=30, ha="right")
            ax.set_ylabel(f"Mean {_METRIC_LABELS[metric]}")
            ax.set_title(f"{field}: {_METRIC_LABELS[metric]} (labels are n)")
            ax.legend(fontsize=8)
    _save(fig, path)
    return True


def _plot_pi(rows: Sequence[Mapping[str, object]], path: Path) -> bool:
    pi = _values(rows, "pi")
    if not np.isfinite(pi).any():
        return False
    info = _values(rows, "pooled_ip_information_gain_control_per_read")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    finite_pi = pi[np.isfinite(pi)]
    axes[0].hist(finite_pi, bins=min(50, max(10, int(np.sqrt(finite_pi.size)))))
    axes[0].set_xlabel("pi")
    axes[0].set_ylabel("Loci")
    axes[0].set_title("Latent target mixture weight")
    _scatter_or_hexbin(axes[1], pi, info)
    axes[1].set_xlabel("pi")
    axes[1].set_ylabel("IP information gain over control / read")
    axes[1].set_title("Mixture diagnostic")
    _save(fig, path)
    return True


def create_rbpnet_evaluation_plots(
    rows: Sequence[Mapping[str, object]],
    calibration: Sequence[Mapping[str, object]],
    plots_dir: str | Path,
    *,
    replicate_names: Sequence[str],
    representative: Sequence[Mapping[str, object]],
    dataset: RBPNetDataset,
    model: torch.nn.Module,
    device: torch.device,
    saved_profile_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Write the standard RBPNet diagnostic plot collection."""

    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: dict[str, str] = {}

    always = (
        ("profile_performance_vs_read_depth.png", _plot_profile_depth),
        ("profile_metric_distributions.png", _plot_profile_distributions),
    )
    for name, function in always:
        function(rows, out / name)
        written.append(name)

    optional = (
        (
            "eta_vs_empirical_enrichment.png",
            lambda path: _plot_eta(calibration, replicate_names, path),
            "checkpoint has no informative enrichment-head observations",
        ),
        (
            "enrichment_calibration.png",
            lambda path: _plot_calibration(calibration, replicate_names, path),
            "checkpoint has no informative enrichment-head observations",
        ),
        (
            "representative_profile_examples.png",
            lambda path: _plot_representatives(
                representative,
                dataset,
                model,
                device,
                path,
                saved_profile_paths,
            ),
            "no examples have both informative IP and SMInput profiles",
        ),
        (
            "stratified_performance_summaries.png",
            lambda path: _plot_stratified(rows, path),
            "selection_state and region_type metadata are absent",
        ),
        (
            "pi_diagnostics.png",
            lambda path: _plot_pi(rows, path),
            "pi predictions are absent",
        ),
    )
    for name, function, reason in optional:
        path = out / name
        if function(path):
            written.append(name)
        else:
            if path.is_file():
                path.unlink()
            skipped[name] = reason
    return {"written": written, "skipped": skipped}
