"""Shared interpretation backend adapters.

Tangermeme operates on editable one-hot sequence channels and passes any
additional model inputs separately. TranscriptML stores Saluki annotations in
the same tensor as the sequence, so this module provides the single bridge
used by perturbation methods today and gradient attribution methods later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from transcriptml.models.common import squeeze_prediction

InterpretationBackend = Literal["tangermeme", "transcriptml"]


def normalize_backend(backend: str) -> InterpretationBackend:
    """Validate and normalize an interpretation backend name."""

    value = str(backend).strip().lower()
    if value not in {"tangermeme", "transcriptml"}:
        raise ValueError("backend must be 'tangermeme' or 'transcriptml'")
    return value  # type: ignore[return-value]


class _NumpyCallableModule(torch.nn.Module):
    """Expose a legacy NumPy predictor through Tangermeme's model protocol."""

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.eval()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = self.function(X.detach().cpu().numpy())
        return torch.as_tensor(np.asarray(values), dtype=X.dtype, device=X.device)


class _EnsembleModule(torch.nn.Module):
    """Torch module reproducing TranscriptML's scalar ensemble reductions."""

    def __init__(self, models: list[torch.nn.Module], reduction: str):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.reduction = reduction
        self.train(any(model.training for model in models))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        values = torch.stack([squeeze_prediction(model(X)) for model in self.models], dim=0)
        if self.reduction == "mean":
            return values.mean(dim=0)
        ordered = values.sort(dim=0).values
        middle = ordered.shape[0] // 2
        if ordered.shape[0] % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


class _AnnotatedSequenceModel(torch.nn.Module):
    """Recombine editable bases with fixed per-position annotations."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.train(model.training)

    def forward(self, bases: torch.Tensor, annotations: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((bases, annotations), dim=1))


@dataclass(frozen=True)
class TangermemeInput:
    """A predictor and encoded batch prepared for a Tangermeme analysis."""

    model: torch.nn.Module
    sequences: torch.Tensor
    args: tuple[torch.Tensor, ...] | None
    device: torch.device
    batch_size: int
    differentiable: bool


def _predictor_module(predictor) -> tuple[torch.nn.Module, torch.device, int, bool]:
    """Return a torch-compatible module and execution settings for a predictor."""

    # Local imports avoid a module cycle: Predictor itself uses tangermeme.predict.
    from transcriptml.interpret.predictor import EnsemblePredictor, Predictor

    if isinstance(predictor, Predictor):
        if isinstance(predictor.model, torch.nn.Module):
            return predictor.model, predictor.device, predictor.batch_size, True
        return _NumpyCallableModule(predictor.model), torch.device("cpu"), predictor.batch_size, False

    if isinstance(predictor, EnsemblePredictor):
        modules: list[torch.nn.Module] = []
        differentiable = True
        devices: list[torch.device] = []
        batch_sizes: list[int] = []
        for member in predictor.predictors:
            module, device, batch_size, member_differentiable = _predictor_module(member)
            modules.append(module)
            devices.append(device)
            batch_sizes.append(batch_size)
            differentiable &= member_differentiable
        device = devices[0] if differentiable and all(item == devices[0] for item in devices) else torch.device("cpu")
        return _EnsembleModule(modules, predictor.reduction), device, min(batch_sizes), differentiable

    raise TypeError("Tangermeme analyses require a Predictor or EnsemblePredictor")


def _compact_cpu_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return a writable contiguous CPU tensor, using int8 for binary/byte inputs."""

    if isinstance(values, torch.Tensor):
        t = values.detach().cpu()
        if t.dtype in (torch.int8, torch.uint8):
            return t.contiguous()
        if not t.is_floating_point():
            if t.numel() == 0 or (int(t.min().item()) >= -128 and int(t.max().item()) <= 127):
                return t.to(torch.int8).contiguous()
            return t.contiguous()
        if t.numel() == 0 or bool(torch.all((t == 0) | (t == 1))):
            return t.to(torch.int8).contiguous()
        return t.contiguous()

    arr = np.asarray(values)
    if arr.dtype in (np.int8, np.uint8):
        if not arr.flags.writeable or not arr.flags.c_contiguous:
            arr = np.array(arr, copy=True, order="C")
        return torch.as_tensor(arr)
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        if arr.size == 0 or (int(arr.min()) >= -128 and int(arr.max()) <= 127):
            return torch.as_tensor(np.array(arr, dtype=np.int8, copy=True, order="C"))
        return torch.as_tensor(np.array(arr, copy=True, order="C"))
    if arr.size == 0 or bool(np.all((arr == 0) | (arr == 1))):
        return torch.as_tensor(np.array(arr, dtype=np.int8, copy=True, order="C"))
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.array(arr, copy=True, order="C")
    return torch.as_tensor(arr)


def prepare_tangermeme_input(
    predictor,
    X: np.ndarray | torch.Tensor,
    *,
    batch_size: int | None = None,
    separate_annotations: bool = True,
) -> TangermemeInput:
    """Prepare sequence channels and fixed annotations for Tangermeme.

    When ``separate_annotations`` is true, only the first four channels are
    mutable. Remaining channels are passed as aligned model arguments and are
    concatenated immediately before the underlying TranscriptML model runs.
    """

    module, device, default_batch_size, differentiable = _predictor_module(predictor)
    shape = tuple(X.shape) if isinstance(X, torch.Tensor) else np.asarray(X).shape
    if len(shape) != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {shape}")

    args: tuple[torch.Tensor, ...] | None = None
    if separate_annotations:
        if shape[1] < 4:
            raise ValueError(f"Expected at least four nucleotide channels, got {shape[1]}")
        values = _compact_cpu_tensor(X[:, :4])
        if shape[1] > 4:
            args = (_compact_cpu_tensor(X[:, 4:]),)
            module = _AnnotatedSequenceModel(module)
    else:
        values = _compact_cpu_tensor(X)

    resolved_batch_size = int(default_batch_size if batch_size is None else batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return TangermemeInput(
        model=module,
        sequences=values,
        args=args,
        device=device,
        batch_size=resolved_batch_size,
        differentiable=differentiable,
    )
