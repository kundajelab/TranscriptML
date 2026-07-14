#!/usr/bin/env python3
"""Interactive MPRA single-nucleotide ISM visualization workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from transcriptml.plotting.single_nt_ism import load_array, plot_single_nt_ism, resolve_sequence_index


# %% CONFIG: edit these values before running

ISM_PATH = Path("__EDIT_ME_ISM_ARRAY__")  # e.g. average_mean_centered_deltas.npy
ISM_KEY = None

SEQ_FEATURES_PATH = Path("__EDIT_ME_X_NPY__")  # Optional; set to None for heatmap only.
SEQ_FEATURES_KEY = None

SEQ_INDEX = 0
START = 0
END = None
SHOW_LOGO = True
BASE_LABELS = "A,C,G,U"

SAVE_FIGURE = False
OUT_PATH = Path("single_nt_ism_mpra.png")


# %% Load arrays

ism = load_array(ISM_PATH, ISM_KEY)
seq_features = None if SEQ_FEATURES_PATH is None else load_array(SEQ_FEATURES_PATH, SEQ_FEATURES_KEY)

print(f"Loaded ISM shape: {ism.shape}")
if seq_features is not None:
    print(f"Loaded sequence/features shape: {seq_features.shape}")


# %% Resolve and plot

seq_index, metadata_record = resolve_sequence_index(n_sequences=ism.shape[0], seq_index=SEQ_INDEX)

fig, axes = plot_single_nt_ism(
    ism,
    seq_index,
    seq_features=seq_features,
    metadata_record=metadata_record,
    start=START,
    end=END,
    base_labels=BASE_LABELS,
    show_logo=SHOW_LOGO,
)

plt.show()

if SAVE_FIGURE:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")
