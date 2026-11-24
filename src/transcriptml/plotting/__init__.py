"""Plotting utilities for TranscriptML analyses."""

from __future__ import annotations

import importlib

__all__ = [
    "plot_single_nt_ism",
    "plot_training_curve",
    "plot_training_history",
    "plot_training_loss",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "plot_single_nt_ism": ("transcriptml.plotting.single_nt_ism", "plot_single_nt_ism"),
    "plot_training_curve": ("transcriptml.plotting.training", "plot_training_curve"),
    "plot_training_history": ("transcriptml.plotting.training", "plot_training_history"),
    "plot_training_loss": ("transcriptml.plotting.training", "plot_training_loss"),
}


def __getattr__(name: str):
    """Lazily load plotting utilities so training startup does not import Matplotlib."""

    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'transcriptml.plotting' has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
