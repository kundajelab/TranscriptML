from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from transcriptml.models.common import ChannelLayerNorm, dropout_or_identity


@dataclass
class SalukiLikeConfig:
    in_ch: int = 6
    base_ch: int = 64
    kernel_size: int = 5
    n_convs: int = 4
    pool_size: int = 2
    dropout: float = 0.2
    gru_hidden: int = 64
    gru_layers: int = 1
    bidirectional: bool = False
    head_hidden: int = 64
    output_dim: int = 1

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``SalukiLike``."""

        return asdict(self)


class SalukiLike(nn.Module):
    """A minimal Saluki-inspired Conv/GRU model.

    This keeps the useful shape of the legacy Saluki models without attempting
    to reproduce every experimental branch.
    """

    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = 64,
        kernel_size: int = 5,
        n_convs: int = 4,
        pool_size: int = 2,
        dropout: float = 0.2,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        bidirectional: bool = False,
        head_hidden: int = 64,
        output_dim: int = 1,
    ):
        """Create the Saluki-inspired Conv/GRU regression model."""

        super().__init__()
        if n_convs < 1:
            raise ValueError("n_convs must be at least 1")
        pad = int(kernel_size) // 2
        blocks: list[nn.Module] = []
        ch = int(in_ch)
        for _ in range(int(n_convs)):
            blocks.extend(
                [
                    nn.Conv1d(ch, int(base_ch), kernel_size=int(kernel_size), padding=pad),
                    ChannelLayerNorm(int(base_ch)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.MaxPool1d(int(pool_size)) if int(pool_size) > 1 else nn.Identity(),
                ]
            )
            ch = int(base_ch)
        self.encoder = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=int(base_ch),
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
        )
        out_ch = int(gru_hidden) * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_ch, int(head_hidden)),
            nn.ReLU(),
            dropout_or_identity(float(dropout)),
            nn.Linear(int(head_hidden), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on ``(B, C, L)`` Saluki-style inputs."""

        z = self.encoder(x.float()).transpose(1, 2)
        _, h_n = self.gru(z)
        if self.gru.bidirectional:
            emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            emb = h_n[-1]
        return self.head(emb).squeeze(-1)
