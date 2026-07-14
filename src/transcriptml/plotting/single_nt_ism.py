#!/usr/bin/env python3
"""Visualize single-nucleotide ISM arrays for one transcript sequence.

The core input is an ISM array with shape (N, 4, L), where channels are
interpreted as A, C, G, U by default. An optional sequence/features array with
shape (N, 4, L) or (N, 6, L) adds a contribution-scaled sequence-logo track and,
for six-channel inputs, a compact transcript isoform diagram. Use --no-logo
or show_logo=False for long regions where per-position letters are too dense.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-single-nt-ism"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib

if (
    "matplotlib.pyplot" not in sys.modules
    and "MPLBACKEND" not in os.environ
    and not os.environ.get("DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


DEFAULT_BASE_LABELS = ("A", "C", "G", "U")
DEFAULT_BASE_COLORS = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "U": "#d62728",
    "T": "#d62728",
}


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    """Load a .npy or .npz array, requiring a key for ambiguous .npz files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Array file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        if key is not None:
            raise ValueError(f"`key` was provided for .npy file {path}; keys only apply to .npz files.")
        return np.load(path, allow_pickle=False, mmap_mode="r")

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            if key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"{path} contains multiple arrays {keys}; provide the desired key."
                    )
                key = keys[0]
            if key not in data:
                raise KeyError(f"{path} does not contain key {key!r}. Available keys: {keys}")
            return np.asarray(data[key])

    raise ValueError(f"Unsupported array file suffix {suffix!r}; expected .npy or .npz.")


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load list-of-records metadata where list position corresponds to sequence index."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, list) or not all(isinstance(record, dict) for record in metadata):
        raise ValueError(
            f"{path} must be a JSON list of objects, with one object per sequence index."
        )

    return metadata


def resolve_sequence_index(
    *,
    n_sequences: int,
    seq_index: int | None = None,
    metadata: list[dict[str, Any]] | None = None,
    metadata_field: str | None = None,
    metadata_value: str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Resolve one sequence index directly or by matching a unique metadata value."""
    using_metadata_lookup = metadata_field is not None or metadata_value is not None

    if seq_index is not None and using_metadata_lookup:
        raise ValueError("Provide either --seq-index or a metadata lookup, not both.")
    if seq_index is None and not using_metadata_lookup:
        raise ValueError("Provide --seq-index, --gene-id, or --metadata-field/--metadata-value.")
    if using_metadata_lookup and metadata is None:
        raise ValueError("Metadata lookup requires --metadata.")
    if (metadata_field is None) != (metadata_value is None):
        raise ValueError("Metadata lookup requires both --metadata-field and --metadata-value.")

    if metadata is not None and len(metadata) != n_sequences:
        raise ValueError(
            f"Metadata has {len(metadata)} records, but the ISM array has N={n_sequences} sequences."
        )

    if seq_index is not None:
        if not 0 <= seq_index < n_sequences:
            raise IndexError(f"seq_index {seq_index} out of range for N={n_sequences}.")
        record = metadata[seq_index] if metadata is not None else None
        return seq_index, record

    assert metadata is not None
    assert metadata_field is not None
    assert metadata_value is not None

    matches = [
        idx
        for idx, record in enumerate(metadata)
        if metadata_field in record and str(record[metadata_field]) == str(metadata_value)
    ]
    if not matches:
        raise KeyError(
            f"No metadata record matched {metadata_field!r} == {metadata_value!r}."
        )
    if len(matches) > 1:
        preview = ", ".join(str(idx) for idx in matches[:8])
        if len(matches) > 8:
            preview += ", ..."
        raise ValueError(
            f"Metadata lookup {metadata_field!r} == {metadata_value!r} matched "
            f"{len(matches)} sequences: {preview}. Use --seq-index or a unique field."
        )

    idx = matches[0]
    return idx, metadata[idx]


def parse_base_labels(labels: str | Iterable[str]) -> tuple[str, str, str, str]:
    """Parse base labels from a comma-delimited string or iterable."""
    if isinstance(labels, str):
        parsed = tuple(part.strip() for part in labels.split(",") if part.strip())
    else:
        parsed = tuple(str(part) for part in labels)
    if len(parsed) != 4:
        raise ValueError(f"Expected exactly four base labels; got {parsed!r}.")
    return parsed  # type: ignore[return-value]


def _base_colors(base_labels: tuple[str, str, str, str]) -> list[str]:
    fallback = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    return [
        DEFAULT_BASE_COLORS.get(base.upper(), fallback[idx])
        for idx, base in enumerate(base_labels)
    ]


def _validate_inputs(
    ism: np.ndarray,
    seq_features: np.ndarray | None,
    seq_index: int,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    ism = np.asarray(ism)
    if ism.ndim != 3:
        raise ValueError(f"`ism` must have shape (N, 4, L); got {ism.shape}.")
    if ism.shape[1] != 4:
        raise ValueError(f"`ism` must have exactly 4 nucleotide channels; got {ism.shape}.")

    n_sequences, _, length = ism.shape
    if not 0 <= seq_index < n_sequences:
        raise IndexError(f"seq_index {seq_index} out of range for N={n_sequences}.")

    if seq_features is None:
        return ism, None, n_sequences, length

    seq_features = np.asarray(seq_features)
    if seq_features.ndim != 3:
        raise ValueError(
            f"`seq_features` must have shape (N, 4, L) or (N, 6, L); got {seq_features.shape}."
        )
    if seq_features.shape[1] not in (4, 6):
        raise ValueError(
            f"`seq_features` must have 4 or 6 channels; got {seq_features.shape}."
        )
    if seq_features.shape[0] != n_sequences or seq_features.shape[2] != length:
        raise ValueError(
            "`seq_features` must match `ism` in N and L. "
            f"Got seq_features={seq_features.shape}, ism={ism.shape}."
        )

    return ism, seq_features, n_sequences, length


def _resolve_slice(
    length: int,
    start: int,
    end: int | None,
) -> tuple[int, int]:
    if end is None:
        end = length
    if not 0 <= start < end <= length:
        raise ValueError(f"Invalid slice start={start}, end={end}, L={length}.")
    return start, end


def _trim_right_padding(
    *,
    region: np.ndarray,
    seq_onehot: np.ndarray | None,
    start: int,
    end: int,
    pad_atol: float,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if seq_onehot is None:
        return region, None, end

    pad_mask = np.all(np.isclose(seq_onehot, 0.0, atol=pad_atol, rtol=0.0), axis=0)
    if not pad_mask.any():
        return region, seq_onehot, end

    first_pad = int(np.argmax(pad_mask))
    region = region[:, :first_pad]
    seq_onehot = seq_onehot[:, :first_pad]
    end = start + first_pad

    if region.shape[1] == 0:
        raise ValueError("Selected region is empty after trimming padding.")

    return region, seq_onehot, end


def _auto_limits(
    values: np.ndarray,
    *,
    center: float,
    symmetric: bool,
    robust: bool,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Selected region contains no finite ISM values.")

    if symmetric and vmin is None and vmax is None:
        shifted_abs = np.abs(finite - center)
        amp = float(np.nanquantile(shifted_abs, 0.98)) if robust else float(np.nanmax(shifted_abs))
        if not np.isfinite(amp) or amp == 0:
            amp = 1.0
        return center - amp, center + amp

    if vmin is None:
        vmin = float(np.nanquantile(finite, 0.02)) if robust else float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanquantile(finite, 0.98)) if robust else float(np.nanmax(finite))
    if vmin == vmax:
        delta = abs(vmin) if vmin != 0 else 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def _observed_base_indices(seq_onehot: np.ndarray, *, atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    active = np.any(np.abs(seq_onehot) > atol, axis=0)
    base_idx = np.argmax(seq_onehot, axis=0)
    return base_idx, active


def _draw_scaled_letter(
    ax: Axes,
    *,
    letter: str,
    x_center: float,
    value: float,
    color: str,
    font_prop: FontProperties,
    max_width: float = 0.82,
) -> None:
    height = abs(float(value))
    if height == 0 or not np.isfinite(height):
        return

    text_path = TextPath((0, 0), letter, size=1, prop=font_prop)
    bounds = text_path.get_extents()
    if bounds.width == 0 or bounds.height == 0:
        return

    normalize = Affine2D().translate(
        -bounds.x0 - bounds.width / 2.0,
        -bounds.y0,
    )
    scale_y = height / bounds.height
    if value < 0:
        scale_y *= -1
    transform = (
        normalize
        + Affine2D().scale(max_width / bounds.width, scale_y)
        + Affine2D().translate(x_center, 0)
        + ax.transData
    )
    patch = PathPatch(
        text_path,
        facecolor=color,
        edgecolor="none",
        transform=transform,
        clip_on=True,
    )
    ax.add_patch(patch)


def _draw_logo(
    ax: Axes,
    *,
    seq_onehot: np.ndarray,
    ism_region: np.ndarray,
    start: int,
    base_labels: tuple[str, str, str, str],
    base_colors: list[str],
    logo_vlim: float | None,
    logo_ylim: float,
    robust: bool,
    fallback_amp: float,
    font_family: str,
) -> None:
    base_idx, active = _observed_base_indices(seq_onehot)
    positions = np.arange(seq_onehot.shape[1])
    observed_values = ism_region[base_idx, positions]
    observed_values = np.where(active, observed_values, np.nan)
    finite_observed = observed_values[np.isfinite(observed_values)]

    if logo_vlim is None:
        if finite_observed.size:
            amp = (
                float(np.nanquantile(np.abs(finite_observed), 0.98))
                if robust
                else float(np.nanmax(np.abs(finite_observed)))
            )
        else:
            amp = fallback_amp
        if not np.isfinite(amp) or amp == 0:
            amp = fallback_amp if fallback_amp != 0 else 1.0
    else:
        amp = abs(float(logo_vlim))
        if amp == 0:
            amp = fallback_amp if fallback_amp != 0 else 1.0

    font_prop = FontProperties(family=font_family, weight="bold")
    for rel_pos, idx, is_active, value in zip(positions, base_idx, active, observed_values):
        if not is_active or not np.isfinite(value):
            continue
        _draw_scaled_letter(
            ax,
            letter=base_labels[int(idx)],
            x_center=start + int(rel_pos) + 0.5,
            value=float(value),
            color=base_colors[int(idx)],
            font_prop=font_prop,
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.65)
    ax.set_xlim(start, start + seq_onehot.shape[1])
    ax.set_ylim(-amp * logo_ylim, amp * logo_ylim)
    ax.set_ylabel("Ref\nISM", rotation=0, ha="right", va="center", labelpad=20)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8, length=2, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _codon_starts_to_cds_mask(codon_start_channel: np.ndarray) -> np.ndarray:
    cds_mask = np.zeros(codon_start_channel.shape[0], dtype=bool)
    for codon_start in np.flatnonzero(codon_start_channel > 0.5):
        cds_mask[codon_start : min(codon_start + 3, cds_mask.shape[0])] = True
    return cds_mask


def _mask_to_intervals(mask: np.ndarray, *, offset: int) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    return [(offset + int(start), offset + int(end)) for start, end in zip(starts, ends)]


def _draw_isoform(
    ax: Axes,
    *,
    seq_features_full: np.ndarray,
    start: int,
    end: int,
    gene_name: str | None,
) -> None:
    cds_mask_full = _codon_starts_to_cds_mask(seq_features_full[4, :])
    cds_mask = cds_mask_full[start:end]
    splice_mask = seq_features_full[5, start:end] > 0.5

    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if end > start:
        ax.add_patch(
            Rectangle(
                (start, 0.43),
                end - start,
                0.16,
                facecolor="black",
                edgecolor="none",
                alpha=0.72,
            )
        )

    for cds_start, cds_end in _mask_to_intervals(cds_mask, offset=start):
        ax.add_patch(
            Rectangle(
                (cds_start, 0.31),
                cds_end - cds_start,
                0.40,
                facecolor="black",
                edgecolor="none",
                alpha=0.92,
            )
        )

    for rel_pos in np.flatnonzero(splice_mask):
        x = start + int(rel_pos) + 1
        ax.vlines(x, ymin=0.16, ymax=0.88, color="#d62728", linewidth=1.2, alpha=0.9)

    if gene_name:
        ax.text(
            -0.01,
            0.5,
            gene_name,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )


def _default_figsize(width: int, *, show_logo: bool, show_isoform: bool) -> tuple[float, float]:
    fig_width = min(24.0, max(8.0, 3.5 + width * 0.065))
    fig_height = 2.4
    if show_logo:
        fig_height += 1.05
    if show_isoform:
        fig_height += 0.65
    return fig_width, fig_height


def _metadata_title(record: dict[str, Any] | None, seq_index: int) -> str:
    if not record:
        return f"Sequence {seq_index}"
    if "gene_id" in record:
        return f"{record['gene_id']} (sequence {seq_index})"
    return f"Sequence {seq_index}"


def plot_single_nt_ism(
    ism: np.ndarray,
    seq_index: int,
    *,
    seq_features: np.ndarray | None = None,
    start: int = 0,
    end: int | None = None,
    base_labels: str | Iterable[str] = DEFAULT_BASE_LABELS,
    title: str | None = None,
    gene_name: str | None = None,
    metadata_record: dict[str, Any] | None = None,
    trim_padding: bool = True,
    pad_atol: float = 0.0,
    show_logo: bool = True,
    show_isoform: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    robust: bool = True,
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    logo_vlim: float | None = None,
    logo_ylim: float = 1.08,
    logo_font: str = "DejaVu Sans",
    show_cbar: bool = True,
    cbar_label: str = "ISM value",
    show_xticks: bool = True,
    max_xticks: int = 12,
    xtick_rotation: int = 0,
    mean_center: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, Axes | None]]:
    """Plot ISM heatmap, optional sequence logo, and optional isoform diagram.

    Parameters
    ----------
    ism
        Array with shape (N, 4, L). Rows are nucleotide channels.
    seq_index
        Sequence index to plot.
    seq_features
        Optional array with shape (N, 4, L) or (N, 6, L). The first four
        channels must be one-hot nucleotide sequence. If six channels are
        supplied, channel 4 marks codon starts and channel 5 marks 5' splice
        sites.
    start, end
        Zero-based, end-exclusive region to plot.
    show_logo
        If False, suppress the sequence-logo track even when sequence features
        are provided. The isoform diagram can still be drawn from six-channel
        sequence features.
    """
    base_labels = parse_base_labels(base_labels)
    colors = _base_colors(base_labels)

    ism, seq_features, _, length = _validate_inputs(ism, seq_features, seq_index)
    start, end = _resolve_slice(length, start, end)

    seq_onehot = None
    seq_features_full = None
    if seq_features is not None:
        seq_features_full = seq_features[seq_index, :, :]
        seq_onehot = seq_features_full[:4, start:end]

    region = np.asarray(ism[seq_index, :, start:end], dtype=float)
    if trim_padding and seq_onehot is not None:
        region, seq_onehot, end = _trim_right_padding(
            region=region,
            seq_onehot=seq_onehot,
            start=start,
            end=end,
            pad_atol=pad_atol,
        )

    if region.shape[1] == 0:
        raise ValueError("Selected region is empty.")

    if mean_center:
        region = region - np.nanmean(region, axis=0, keepdims=True)

    width = int(region.shape[1])
    plot_end = start + width
    hm_vmin, hm_vmax = _auto_limits(
        region,
        center=center,
        symmetric=symmetric,
        robust=robust,
        vmin=vmin,
        vmax=vmax,
    )
    fallback_amp = max(abs(hm_vmin - center), abs(hm_vmax - center), 1.0)

    want_logo = show_logo and seq_onehot is not None
    want_isoform = (
        show_isoform
        and seq_features_full is not None
        and seq_features_full.shape[0] == 6
    )
    if figsize is None:
        figsize = _default_figsize(width, show_logo=want_logo, show_isoform=want_isoform)

    height_ratios: list[float] = []
    row_names: list[str] = []
    if want_logo:
        height_ratios.append(0.48)
        row_names.append("logo")
    height_ratios.append(1.0)
    row_names.append("heatmap")
    if want_isoform:
        height_ratios.append(0.35)
        row_names.append("isoform")

    fig = plt.figure(figsize=figsize)
    width_ratios = [1.0, 0.035] if show_cbar else [1.0, 0.001]
    gs = fig.add_gridspec(
        nrows=len(row_names),
        ncols=2,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.08,
        wspace=0.06,
    )

    axes: dict[str, Axes | None] = {
        "logo": None,
        "heatmap": None,
        "isoform": None,
        "colorbar": None,
    }
    row_lookup = {name: idx for idx, name in enumerate(row_names)}

    if want_logo:
        ax_logo = fig.add_subplot(gs[row_lookup["logo"], 0])
        _draw_logo(
            ax_logo,
            seq_onehot=seq_onehot,
            ism_region=region,
            start=start,
            base_labels=base_labels,
            base_colors=colors,
            logo_vlim=logo_vlim,
            logo_ylim=logo_ylim,
            robust=robust,
            fallback_amp=fallback_amp,
            font_family=logo_font,
        )
        axes["logo"] = ax_logo

    ax_hm = fig.add_subplot(gs[row_lookup["heatmap"], 0], sharex=axes["logo"])
    norm = (
        TwoSlopeNorm(vmin=hm_vmin, vcenter=center, vmax=hm_vmax)
        if hm_vmin < center < hm_vmax
        else None
    )
    im = ax_hm.imshow(
        region,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else hm_vmin,
        vmax=None if norm is not None else hm_vmax,
        interpolation="nearest",
        extent=(start, plot_end, 4, 0),
    )
    ax_hm.set_xlim(start, plot_end)
    ax_hm.set_yticks(np.arange(4) + 0.5)
    ax_hm.set_yticklabels(base_labels)
    ax_hm.set_ylabel("Mutation")
    ax_hm.tick_params(axis="y", length=0)
    axes["heatmap"] = ax_hm

    tick_positions: np.ndarray | None = None
    tick_labels: list[str] | None = None
    if show_xticks:
        if max_xticks < 2:
            max_xticks = 2
        abs_left = start
        abs_right = plot_end - 1
        if width <= 1:
            tick_abs = np.array([abs_left], dtype=int)
        else:
            tick_abs = np.linspace(abs_left, abs_right, num=min(max_xticks, width), dtype=int)
            tick_abs = np.unique(tick_abs)
        tick_positions = tick_abs + 0.5
        tick_labels = [str(tick) for tick in tick_abs]
    else:
        ax_hm.set_xticks([])

    if show_xticks and not want_isoform:
        assert tick_positions is not None
        assert tick_labels is not None
        ax_hm.set_xticks(tick_positions)
        ax_hm.set_xticklabels(tick_labels, rotation=xtick_rotation)
        ax_hm.tick_params(axis="x", length=3, pad=2)
        ax_hm.set_xlabel("Position")
    else:
        ax_hm.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_hm.set_xlabel("")

    if show_cbar:
        cbar_rows = slice(0, row_lookup["heatmap"] + 1)
        ax_cbar = fig.add_subplot(gs[cbar_rows, 1])
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label(cbar_label)
        axes["colorbar"] = ax_cbar

    if want_isoform:
        ax_iso = fig.add_subplot(gs[row_lookup["isoform"], 0], sharex=ax_hm)
        _draw_isoform(
            ax_iso,
            seq_features_full=seq_features_full,
            start=start,
            end=plot_end,
            gene_name=gene_name,
        )
        axes["isoform"] = ax_iso
        if show_xticks:
            assert tick_positions is not None
            assert tick_labels is not None
            ax_iso.set_xticks(tick_positions)
            ax_iso.set_xticklabels(tick_labels, rotation=xtick_rotation)
            ax_iso.tick_params(axis="x", length=3, pad=2)
            ax_iso.set_xlabel("Position")
        else:
            ax_iso.set_xticks([])

    if title is None:
        title = _metadata_title(metadata_record, seq_index)
    if title:
        fig.suptitle(title, y=0.99, fontsize=12)

    fig.align_ylabels([ax for ax in axes.values() if ax is not None])
    return fig, axes


def make_demo_arrays() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Create small synthetic inputs for CLI smoke tests and examples."""
    rng = np.random.default_rng(7)
    n_sequences, length = 2, 96
    seq_features = np.zeros((n_sequences, 6, length), dtype=float)
    bases = rng.integers(0, 4, size=(n_sequences, length))
    for seq_idx in range(n_sequences):
        seq_features[seq_idx, bases[seq_idx], np.arange(length)] = 1.0
        seq_features[seq_idx, 4, 12:78:3] = 1.0
        seq_features[seq_idx, 5, [31, 63]] = 1.0

    x = np.linspace(0, 2 * np.pi, length)
    ism = rng.normal(0, 0.22, size=(n_sequences, 4, length))
    ism += np.sin(x)[None, None, :] * np.array([0.18, -0.12, 0.08, -0.16])[None, :, None]
    metadata = [
        {"gene_id": "DEMO1", "chrom": "chrDemo", "strand": "+", "transcript_length": length},
        {"gene_id": "DEMO2", "chrom": "chrDemo", "strand": "-", "transcript_length": length},
    ]
    return ism, seq_features, metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a single sequence from a single-nucleotide ISM array. "
            "The optional sequence/features array adds a sequence-logo track "
            "and, for six-channel arrays, a CDS/splice isoform diagram."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ism", type=Path, help="Path to .npy or .npz ISM array with shape (N, 4, L).")
    parser.add_argument("--ism-key", help="Array key to read from --ism when it is a .npz file.")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional dataset bundle directory used to default X.npy and metadata.json.",
    )
    parser.add_argument(
        "--seq-features",
        type=Path,
        help="Optional .npy or .npz array with shape (N, 4, L) or (N, 6, L).",
    )
    parser.add_argument(
        "--seq-features-key",
        help="Array key to read from --seq-features when it is a .npz file.",
    )
    parser.add_argument("--seq-index", type=int, help="Zero-based sequence index to plot.")
    parser.add_argument("--metadata", type=Path, help="Optional metadata JSON list with one record per sequence.")
    parser.add_argument(
        "--metadata-field",
        help="Metadata field to match for sequence lookup, for example gene_id.",
    )
    parser.add_argument("--metadata-value", help="Metadata value to match for sequence lookup.")
    parser.add_argument("--gene-id", help="Shortcut for --metadata-field gene_id --metadata-value VALUE.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start position, inclusive.")
    parser.add_argument("--end", type=int, help="End position, exclusive.")
    parser.add_argument("--out", type=Path, help="Output figure path, such as .png, .pdf, or .svg.")
    parser.add_argument("--dpi", type=int, default=220, help="DPI used when saving raster output.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after plotting.")
    parser.add_argument("--demo", action="store_true", help="Use built-in synthetic arrays instead of --ism.")

    parser.add_argument("--title", help="Figure title. Defaults to gene_id/sequence index when metadata is available.")
    parser.add_argument("--gene-name", help="Label shown under the isoform diagram.")
    parser.add_argument("--base-labels", default="A,C,G,U", help="Comma-delimited nucleotide labels.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for the heatmap.")
    parser.add_argument("--center", type=float, default=0.0, help="Center value for the diverging heatmap scale.")
    parser.add_argument("--vmin", type=float, help="Manual heatmap minimum.")
    parser.add_argument("--vmax", type=float, help="Manual heatmap maximum.")
    parser.add_argument("--logo-vlim", type=float, help="Symmetric y-limit magnitude for the logo track.")
    parser.add_argument("--logo-font", default="DejaVu Sans", help="Font family used for sequence-logo letters.")
    parser.add_argument("--figsize", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Figure size in inches.")
    parser.add_argument("--max-xticks", type=int, default=12, help="Maximum number of x-axis tick labels.")
    parser.add_argument("--xtick-rotation", type=int, default=0, help="Rotation for x-axis tick labels.")
    parser.add_argument("--pad-atol", type=float, default=0.0, help="Tolerance for detecting zero-padded sequence columns.")
    parser.add_argument("--mean-center", action="store_true", help="Subtract the per-position mean across 4 channels.")
    parser.add_argument("--no-trim-padding", action="store_true", help="Do not trim right-side all-zero sequence padding.")
    parser.add_argument(
        "--no-logo",
        action="store_true",
        help="Do not draw the sequence-logo track; useful for long or cluttered regions.",
    )
    parser.add_argument("--no-isoform", action="store_true", help="Do not draw the isoform diagram.")
    parser.add_argument("--no-cbar", action="store_true", help="Do not draw the heatmap colorbar.")
    parser.add_argument("--no-robust", action="store_true", help="Use full min/max instead of robust quantiles.")
    parser.add_argument("--no-symmetric", action="store_true", help="Do not force symmetric heatmap limits around --center.")
    return parser.parse_args(argv)


def _apply_dataset_defaults(args: argparse.Namespace) -> None:
    dataset = getattr(args, "dataset", None)
    if dataset is None:
        return
    dataset = Path(dataset)
    if getattr(args, "seq_features", None) is None and (dataset / "X.npy").exists():
        args.seq_features = dataset / "X.npy"
    if getattr(args, "metadata", None) is None and (dataset / "metadata.json").exists():
        args.metadata = dataset / "metadata.json"


def plot_ism_from_args(args: argparse.Namespace) -> int:
    """Run the single-nucleotide ISM plotting command from parsed CLI args."""

    if args.gene_id is not None:
        if args.metadata_field is not None or args.metadata_value is not None:
            raise ValueError("--gene-id cannot be combined with --metadata-field/--metadata-value.")
        args.metadata_field = "gene_id"
        args.metadata_value = args.gene_id

    if args.demo:
        ism, seq_features, metadata = make_demo_arrays()
        if args.seq_index is None and args.metadata_value is None:
            args.metadata_field = "gene_id"
            args.metadata_value = "DEMO1"
    else:
        _apply_dataset_defaults(args)
        if args.ism is None:
            raise ValueError("--ism is required unless --demo is used.")
        ism = load_array(args.ism, args.ism_key)
        seq_features = (
            load_array(args.seq_features, args.seq_features_key)
            if args.seq_features is not None
            else None
        )
        metadata = load_metadata(args.metadata) if args.metadata is not None else None

    n_sequences = int(np.asarray(ism).shape[0])
    seq_index, metadata_record = resolve_sequence_index(
        n_sequences=n_sequences,
        seq_index=args.seq_index,
        metadata=metadata,
        metadata_field=args.metadata_field,
        metadata_value=args.metadata_value,
    )

    gene_name = args.gene_name
    if gene_name is None and metadata_record is not None and "gene_id" in metadata_record:
        gene_name = str(metadata_record["gene_id"])

    fig, _ = plot_single_nt_ism(
        ism,
        seq_index,
        seq_features=seq_features,
        start=args.start,
        end=args.end,
        base_labels=args.base_labels,
        title=args.title,
        gene_name=gene_name,
        metadata_record=metadata_record,
        trim_padding=not args.no_trim_padding,
        pad_atol=args.pad_atol,
        show_logo=not args.no_logo,
        show_isoform=not args.no_isoform,
        cmap=args.cmap,
        center=args.center,
        robust=not args.no_robust,
        symmetric=not args.no_symmetric,
        vmin=args.vmin,
        vmax=args.vmax,
        logo_vlim=args.logo_vlim,
        logo_font=args.logo_font,
        show_cbar=not args.no_cbar,
        show_xticks=True,
        max_xticks=args.max_xticks,
        xtick_rotation=args.xtick_rotation,
        mean_center=args.mean_center,
        figsize=tuple(args.figsize) if args.figsize is not None else None,
    )

    if args.out is None and not args.show:
        raise ValueError("Provide --out or use --show.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {args.out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    return plot_ism_from_args(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
