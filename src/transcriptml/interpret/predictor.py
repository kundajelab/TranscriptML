from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

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
        self.model = model
        self.device = torch.device(device)
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
    ) -> "Predictor":
        model, _ = load_checkpoint(checkpoint_path, map_location=device)
        return cls(model, device=device, batch_size=batch_size)

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        if not isinstance(self.model, torch.nn.Module):
            return np.asarray(self.model(np.asarray(X)), dtype=np.float32).reshape(-1)
        bs = int(batch_size or self.batch_size)
        outs: list[np.ndarray] = []
        n = int(X.shape[0])
        for start in range(0, n, bs):
            batch = X[start : start + bs]
            if isinstance(batch, torch.Tensor):
                xb = batch.to(self.device, dtype=torch.float32)
            else:
                xb = torch.as_tensor(np.asarray(batch), dtype=torch.float32).to(self.device)
            y = squeeze_prediction(self.model(xb))
            outs.append(y.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1))
        return np.concatenate(outs) if outs else np.empty((0,), dtype=np.float32)


class EnsemblePredictor:
    """Mean or median reduction over multiple predictors."""

    def __init__(self, predictors: Sequence[Predictor], *, reduction: str = "mean"):
        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one predictor")
        if reduction not in {"mean", "median"}:
            raise ValueError("reduction must be 'mean' or 'median'")
        self.predictors = list(predictors)
        self.reduction = reduction

    def predict(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        preds = np.stack([p.predict(X, batch_size=batch_size) for p in self.predictors], axis=0)
        if self.reduction == "mean":
            return preds.mean(axis=0, dtype=np.float64).astype(np.float32)
        return np.median(preds, axis=0).astype(np.float32)
