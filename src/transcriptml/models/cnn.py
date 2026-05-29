from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from transcriptml.models.common import ChannelLayerNorm, dropout_or_identity


@dataclass
class SmallCNNConfig:
    in_ch: int = 4
    n_filters: int = 64
    kernel_size: int = 9
    n_layers: int = 2
    dropout: float = 0.1
    head_hidden: int = 64
    output_dim: int = 1

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``SmallCNN``."""

        return asdict(self)


class SmallCNN(nn.Module):
    """A compact sequence CNN baseline with global max/mean pooling."""

    def __init__(
        self,
        in_ch: int = 4,
        n_filters: int = 64,
        kernel_size: int = 9,
        n_layers: int = 2,
        dropout: float = 0.1,
        head_hidden: int = 64,
        output_dim: int = 1,
    ):
        """Create the compact CNN regression model.

        Args:
            in_ch: Number of input channels in encoded sequence tensors.
            n_filters: Number of convolutional filters in each encoder layer.
            kernel_size: Width of each 1D convolution kernel.
            n_layers: Number of convolutional encoder layers.
            dropout: Dropout probability used in encoder and head layers.
            head_hidden: Hidden dimension of the regression head.
            output_dim: Number of output units produced by the head.
        """

        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        layers: list[nn.Module] = []
        ch = int(in_ch)
        pad = int(kernel_size) // 2
        for _ in range(int(n_layers)):
            layers.extend(
                [
                    nn.Conv1d(ch, int(n_filters), kernel_size=int(kernel_size), padding=pad),
                    ChannelLayerNorm(int(n_filters)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            ch = int(n_filters)
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(int(n_filters) * 2, int(head_hidden)),
            nn.GELU(),
            dropout_or_identity(float(dropout)),
            nn.Linear(int(head_hidden), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on ``(B, C, L)`` encoded sequences.

        Args:
            x: Encoded sequence batch with shape ``(batch, channels, length)``.
        """

        z = self.encoder(x.float())
        pooled = torch.cat([z.amax(dim=-1), z.mean(dim=-1)], dim=1)
        y = self.head(pooled)
        return y.squeeze(-1)
