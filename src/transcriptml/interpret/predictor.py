from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from tangermeme.predict import predict as tangermeme_predict

from transcriptml.devices import resolve_device
from transcriptml.models.common import squeeze_prediction
from transcriptml.models.registry import load_checkpoint


class Predictor:
    """Batched prediction wrapper for a model or callable."""

    def __init__(
        self,
        model: torch.nn.Module | Callable[[np.ndarray], np.ndarray],
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 128,
    ):
        """Create a predictor from a PyTorch module or NumPy callable.

        Args:
            model: PyTorch module or callable that maps an encoded batch to
                scalar predictions.
            device: Torch device used when ``model`` is a PyTorch module.
            batch_size: Default batch size for prediction.
        """

        self.model = model
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        if isinstance(model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 128,
    ) -> Predictor:
        """Load a checkpoint and wrap the reconstructed model for prediction.

        Args:
            checkpoint_path: Path to a TranscriptML checkpoint saved by the
                model registry.
            device: Torch device used to load and run the model.
            batch_size: Default batch size for prediction.
        """

        resolved = resolve_device(device)
        model, _ = load_checkpoint(checkpoint_path, map_location=resolved)
        return cls(model, device=resolved, batch_size=batch_size)

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Predict one scalar output per input sequence.

        Args:
            X: Encoded ``(N, C, L)`` batch as a NumPy array or torch tensor.
            batch_size: Optional batch-size override for this prediction call.
        """

        if not isinstance(self.model, torch.nn.Module):
            return np.asarray(self.model(np.asarray(X)), dtype=np.float32).reshape(-1)
        if int(X.shape[0]) == 0:
            return np.empty((0,), dtype=np.float32)
        bs = int(batch_size or self.batch_size)
        if isinstance(X, torch.Tensor):
            inputs = X.detach()
        else:
            array = np.asarray(X)
            if not array.flags.writeable:
                array = np.array(array, copy=True)
            inputs = torch.as_tensor(array)
        outputs = tangermeme_predict(
            self.model,
            inputs,
            func=squeeze_prediction,
            batch_size=bs,
            dtype=torch.float32,
            device=self.device,
            verbose=False,
        )
        if not isinstance(outputs, torch.Tensor):  # pragma: no cover - scalar contract guards this.
            raise TypeError("predictor model must return one tensor")
        return outputs.numpy().astype(np.float32, copy=False).reshape(-1)


class EnsemblePredictor:
    """Mean or median reduction over multiple predictors."""

    def __init__(self, predictors: Sequence[Predictor], *, reduction: str = "mean"):
        """Create an ensemble predictor with mean or median reduction.

        Args:
            predictors: Non-empty sequence of predictors to evaluate.
            reduction: Reduction to apply across predictors, either ``mean`` or
                ``median``.
        """

        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one predictor")
        if reduction not in {"mean", "median"}:
            raise ValueError("reduction must be 'mean' or 'median'")
        self.predictors = list(predictors)
        self.reduction = reduction

    def predict(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        """Predict with each member and reduce predictions across members.

        Args:
            X: Encoded ``(N, C, L)`` batch as a NumPy array or torch tensor.
            batch_size: Optional batch-size override forwarded to each member
                predictor.
        """

        preds = np.stack([p.predict(X, batch_size=batch_size) for p in self.predictors], axis=0)
        if self.reduction == "mean":
            return preds.mean(axis=0, dtype=np.float64).astype(np.float32)
        return np.median(preds, axis=0).astype(np.float32)
