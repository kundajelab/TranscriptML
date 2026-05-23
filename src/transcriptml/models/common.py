from __future__ import annotations

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for ``(B, C, L)`` tensors."""

    def __init__(self, channels: int, eps: float = 1e-5):
        """Create channel-wise layer normalization."""

        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization while preserving ``(B, C, L)`` layout."""

        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def dropout_or_identity(p: float) -> nn.Module:
    """Return ``Dropout`` when ``p`` is positive, otherwise ``Identity``."""

    return nn.Dropout(float(p)) if p and p > 0 else nn.Identity()


def dropout1d_or_identity(p: float) -> nn.Module:
    """Return ``Dropout1d`` when ``p`` is positive, otherwise ``Identity``."""

    return nn.Dropout1d(float(p)) if p and p > 0 else nn.Identity()


def squeeze_prediction(y: torch.Tensor) -> torch.Tensor:
    """Normalize model output to one scalar prediction per example."""

    if isinstance(y, (tuple, list)):
        y = y[0]
    y = torch.as_tensor(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    if y.ndim > 1:
        flat = y.reshape(y.shape[0], -1)
        if flat.shape[1] == 1:
            return flat[:, 0]
        raise ValueError(f"Expected scalar model output per example, got {tuple(y.shape)}")
    return y
