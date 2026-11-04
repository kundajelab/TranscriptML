"""Training history loss and metric curve plotting utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-transcriptml-training"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib

if (
    "matplotlib.pyplot" not in sys.modules
    and "MPLBACKEND" not in os.environ
    and not os.environ.get("DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _extract_series(
    history: Sequence[Mapping[str, Any]],
    key: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite ``(epochs, values)`` arrays for ``key`` from ``history``."""

    if not key or not history:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    epochs = np.asarray(
        [row.get("epoch", index + 1) for index, row in enumerate(history)],
        dtype=np.float64,
    )
    values = np.asarray(
        [
            np.nan if row.get(key) is None else float(row.get(key, np.nan))
            for row in history
        ],
        dtype=np.float64,
    )
    keep = np.isfinite(epochs) & np.isfinite(values)
    return epochs[keep], values[keep]


def _has_finite_values(
    history: Sequence[Mapping[str, Any]],
    *keys: str,
) -> bool:
    """Return whether any of ``keys`` has at least one finite entry in ``history``."""

    for key in keys:
        _, values = _extract_series(history, key)
        if values.size > 0:
            return True
    return False


def plot_training_curve(
    history: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    train_key: str = "train_loss",
    val_key: str | None = "val_loss",
    ylabel: str = "Loss",
    title: str | None = None,
    best_epoch: int | None = None,
) -> Path:
    """Plot a training and (when finite) validation metric curve across epochs.

    Args:
        history: Per-epoch metric dictionaries containing ``"epoch"`` and metric keys.
        out_path: Destination image path.
        train_key: Key in ``history`` for the training split curve.
        val_key: Optional key in ``history`` for the validation split curve.
        ylabel: Y-axis label.
        title: Optional plot title; defaults to ``ylabel``.
        best_epoch: Optional 1-based best checkpoint epoch marked with a vertical line.
    """

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    train_epochs, train_values = _extract_series(history, train_key)
    val_epochs, val_values = _extract_series(history, val_key)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False

    if train_values.size > 0:
        marker = "o" if train_values.size <= 25 else None
        ax.plot(
            train_epochs,
            train_values,
            label="Train",
            color="#1f77b4",
            linewidth=1.8,
            marker=marker,
            markersize=4.5,
        )
        plotted = True

    if val_values.size > 0:
        marker = "o" if val_values.size <= 25 else None
        ax.plot(
            val_epochs,
            val_values,
            label="Validation",
            color="#ff7f0e",
            linewidth=1.8,
            marker=marker,
            markersize=4.5,
        )
        plotted = True

    if plotted and best_epoch is not None and int(best_epoch) > 0:
        ax.axvline(
            int(best_epoch),
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=f"Best epoch ({int(best_epoch)})",
        )

    if not plotted:
        ax.text(0.5, 0.5, "No finite observations", ha="center", va="center")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else ylabel)

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_training_loss(
    history: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    loss_name: str | None = None,
    best_epoch: int | None = None,
) -> Path:
    """Plot training and validation loss curves across epochs."""

    suffix = f" ({loss_name})" if loss_name else ""
    return plot_training_curve(
        history,
        out_path,
        train_key="train_loss",
        val_key="val_loss",
        ylabel=f"Loss{suffix}",
        title=f"Training and Validation Loss{suffix}",
        best_epoch=best_epoch,
    )


def plot_training_history(
    history: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    loss_name: str | None = None,
    best_epoch: int | None = None,
    include_enrichment: bool | None = None,
) -> dict[str, Path]:
    """Write separate train/validation curve plots for all active metrics in ``history``.

    Always writes ``loss_curve.png``. Writes ``pearson_curve.png`` when finite
    Pearson values exist, and writes RBPNet component loss plots
    (``ip_profile_loss_curve.png``, ``sm_profile_loss_curve.png``, and
    ``enrichment_loss_curve.png`` when enabled) when present in ``history``.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["loss_curve.png"] = plot_training_loss(
        history,
        directory / "loss_curve.png",
        loss_name=loss_name,
        best_epoch=best_epoch,
    )

    if _has_finite_values(history, "train_pearson", "val_pearson"):
        written["pearson_curve.png"] = plot_training_curve(
            history,
            directory / "pearson_curve.png",
            train_key="train_pearson",
            val_key="val_pearson",
            ylabel="Pearson r",
            title="Training and Validation Pearson Correlation",
            best_epoch=best_epoch,
        )

    if _has_finite_values(history, "train_ip_profile_loss", "val_ip_profile_loss"):
        written["ip_profile_loss_curve.png"] = plot_training_curve(
            history,
            directory / "ip_profile_loss_curve.png",
            train_key="train_ip_profile_loss",
            val_key="val_ip_profile_loss",
            ylabel="IP profile loss",
            title="Training and Validation IP Profile Loss",
            best_epoch=best_epoch,
        )

    if _has_finite_values(history, "train_sm_profile_loss", "val_sm_profile_loss"):
        written["sm_profile_loss_curve.png"] = plot_training_curve(
            history,
            directory / "sm_profile_loss_curve.png",
            train_key="train_sm_profile_loss",
            val_key="val_sm_profile_loss",
            ylabel="SMInput profile loss",
            title="Training and Validation SMInput Profile Loss",
            best_epoch=best_epoch,
        )

    if _has_finite_values(history, "train_enrichment_loss", "val_enrichment_loss"):
        if include_enrichment is None:
            _, train_enr = _extract_series(history, "train_enrichment_loss")
            _, val_enr = _extract_series(history, "val_enrichment_loss")
            should_plot_enrichment = bool(
                np.any(train_enr != 0.0) or np.any(val_enr != 0.0)
            )
        else:
            should_plot_enrichment = bool(include_enrichment)
        if should_plot_enrichment:
            written["enrichment_loss_curve.png"] = plot_training_curve(
                history,
                directory / "enrichment_loss_curve.png",
                train_key="train_enrichment_loss",
                val_key="val_enrichment_loss",
                ylabel="Enrichment loss",
                title="Training and Validation Enrichment Loss",
                best_epoch=best_epoch,
            )

    return written
