#!/usr/bin/env python3
"""Interactive Saluki single-nucleotide ISM visualization workflow.

Open this file in an editor that supports Python cells, edit the CONFIG cell,
and run cells one at a time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from transcriptml.plotting.single_nt_ism import (
    load_array,
    load_metadata,
    plot_single_nt_ism,
    resolve_sequence_index,
)


# %% CONFIG: edit these values before running

ISM_PATH = Path("__EDIT_ME_ISM_ARRAY__")  # e.g. average_mean_centered_deltas.npy
ISM_KEY = None

DATASET_DIR = Path("__EDIT_ME_DATASET_DIR__")
SEQ_FEATURES_PATH = DATASET_DIR / "X.npy"
SEQ_FEATURES_KEY = None
METADATA_PATH = DATASET_DIR / "metadata.json"

LOOKUP_FIELD = "gene_id"
LOOKUP_VALUE = "__EDIT_ME_GENE_ID__"
SEQ_INDEX = 0

START = 0
END = None
SHOW_LOGO = False
BASE_LABELS = "A,C,G,U"

SAVE_FIGURE = False
OUT_PATH = Path("single_nt_ism_saluki.png")


# %% Load arrays and metadata

ism = load_array(ISM_PATH, ISM_KEY)
seq_features = load_array(SEQ_FEATURES_PATH, SEQ_FEATURES_KEY)
metadata = load_metadata(METADATA_PATH)

print(f"Loaded ISM shape: {ism.shape}")
print(f"Loaded sequence/features shape: {seq_features.shape}")
print(f"Loaded {len(metadata)} metadata records")


# %% Resolve the sequence index

if LOOKUP_FIELD and LOOKUP_VALUE:
    seq_index, metadata_record = resolve_sequence_index(
        n_sequences=ism.shape[0],
        metadata=metadata,
        metadata_field=LOOKUP_FIELD,
        metadata_value=LOOKUP_VALUE,
    )
else:
    seq_index, metadata_record = resolve_sequence_index(
        n_sequences=ism.shape[0],
        seq_index=SEQ_INDEX,
        metadata=metadata,
    )

print(f"Plotting sequence index: {seq_index}")
print(metadata_record)


# %% Plot one region

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
