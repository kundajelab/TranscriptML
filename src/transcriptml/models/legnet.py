from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from transcriptml.models.common import dropout1d_or_identity, dropout_or_identity


@dataclass
class LegNetConfig:
    in_ch: int = 4
    stem_ch: int = 64
    stem_ks: int = 7
    ef_ks: int = 5
    ef_block_sizes: list[int] = field(default_factory=lambda: [64, 96, 128])
    pool_sizes: list[int] = field(default_factory=lambda: [2, 2, 2])
    resize_factor: int = 4
    block_dropout: float = 0.0
    head_dropout: float = 0.1
    stem_dropout: float = 0.0
    output_dim: int = 1

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``LegNet``."""

        return asdict(self)


class SELayer(nn.Module):
    def __init__(self, inp: int, reduction: int = 4):
        """Create a squeeze-excitation block for 1D sequence channels."""

        super().__init__()
        hidden = max(1, int(inp) // int(reduction))
        self.fc = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, inp), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel reweighting to a ``(B, C, L)`` tensor."""

        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LocalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, dropout: float = 0.0):
        """Create a local convolution, normalization, activation, and dropout block."""

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding="same", bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            dropout1d_or_identity(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the local convolutional block."""

        return self.block(x)


class EffBlock(nn.Module):
    def __init__(self, in_ch: int, ks: int, resize_factor: int, dropout: float = 0.0):
        """Create a LegNet efficient residual-style convolutional block."""

        super().__init__()
        inner = int(in_ch) * int(resize_factor)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, inner, kernel_size=1, bias=False),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            dropout1d_or_identity(dropout),
            nn.Conv1d(inner, inner, kernel_size=ks, padding="same", groups=inner, bias=False),
            nn.BatchNorm1d(inner),
            nn.SiLU(),
            dropout1d_or_identity(dropout),
            SELayer(inner, reduction=max(1, int(resize_factor))),
            nn.Conv1d(inner, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.SiLU(),
            dropout1d_or_identity(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the efficient block."""

        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn: nn.Module):
        """Wrap a module whose output is concatenated with its input."""

        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate transformed and original channels."""

        return torch.cat([self.fn(x), x], dim=1)


class LegNet(nn.Module):
    """A compact LegNet port based on the legacy implementation."""

    def __init__(
        self,
        in_ch: int = 4,
        stem_ch: int = 64,
        stem_ks: int = 7,
        ef_ks: int = 5,
        ef_block_sizes: list[int] | tuple[int, ...] = (64, 96, 128),
        pool_sizes: list[int] | tuple[int, ...] = (2, 2, 2),
        resize_factor: int = 4,
        block_dropout: float = 0.0,
        head_dropout: float = 0.1,
        stem_dropout: float = 0.0,
        output_dim: int = 1,
    ):
        """Create the compact LegNet regression model."""

        super().__init__()
        if len(pool_sizes) != len(ef_block_sizes):
            raise ValueError("pool_sizes and ef_block_sizes must have the same length")
        self.stem = LocalBlock(in_ch, stem_ch, stem_ks, dropout=stem_dropout)
        blocks: list[nn.Module] = []
        ch = int(stem_ch)
        for pool, out_ch in zip(pool_sizes, ef_block_sizes):
            blocks.append(
                nn.Sequential(
                    ResidualConcat(EffBlock(ch, ef_ks, resize_factor, dropout=block_dropout)),
                    LocalBlock(ch * 2, int(out_ch), ef_ks, dropout=block_dropout),
                    nn.MaxPool1d(int(pool)) if int(pool) > 1 else nn.Identity(),
                )
            )
            ch = int(out_ch)
        self.main = nn.Sequential(*blocks)
        self.mapper = nn.Sequential(nn.BatchNorm1d(ch), nn.Conv1d(ch, ch * 2, kernel_size=1))
        self.head = nn.Sequential(
            nn.Linear(ch * 2, ch * 2),
            nn.BatchNorm1d(ch * 2),
            nn.SiLU(),
            dropout_or_identity(head_dropout),
            nn.Linear(ch * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on ``(B, C, L)`` encoded sequences."""

        z = self.stem(x.float())
        z = self.main(z)
        z = self.mapper(z)
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        return self.head(z).squeeze(-1)
