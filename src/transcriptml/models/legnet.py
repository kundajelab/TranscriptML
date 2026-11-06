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
    use_optimal_head: bool = False
    use_optimal_backbone: bool = False

    def to_kwargs(self) -> dict[str, object]:
        """Return constructor keyword arguments for ``LegNet``."""

        return asdict(self)


class SELayer(nn.Module):
    def __init__(self, inp: int, reduction: int = 4):
        """Create a squeeze-excitation block for 1D sequence channels.

        Args:
            inp: Number of input channels to reweight.
            reduction: Channel reduction factor for the hidden excitation layer.
        """

        super().__init__()
        hidden = max(1, int(inp) // int(reduction))
        self.fc = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, inp), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel reweighting to a ``(B, C, L)`` tensor.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.
        """

        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LocalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ks: int,
        dropout: float = 0.0,
        *,
        pool: int = 1,
        use_optimal_backbone: bool = False,
    ):
        """Create a local convolution, normalization, activation, and dropout block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            ks: Convolution kernel size.
            dropout: Dropout probability after activation.
            pool: Max-pooling factor applied before normalization when
                ``use_optimal_backbone=True``.
            use_optimal_backbone: Whether to order the block as ``Conv1d(bias=False)
                -> [MaxPool1d] -> BatchNorm1d -> SiLU -> Dropout1d``.
        """

        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding="same", bias=False),
        ]
        if use_optimal_backbone and int(pool) > 1:
            layers.append(nn.MaxPool1d(int(pool)))
        layers.extend(
            [
                nn.BatchNorm1d(out_ch),
                nn.SiLU(),
                dropout1d_or_identity(dropout),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the local convolutional block.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.
        """

        return self.block(x)


class EffBlock(nn.Module):
    def __init__(self, in_ch: int, ks: int, resize_factor: int, dropout: float = 0.0):
        """Create a LegNet efficient residual-style convolutional block.

        Args:
            in_ch: Number of input and output channels for the block.
            ks: Depthwise convolution kernel size.
            resize_factor: Expansion factor for the internal channel dimension.
            dropout: Dropout probability inside the block.
        """

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
        """Apply the efficient block.

        Args:
            x: Input tensor with shape ``(batch, channels, length)``.
        """

        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn: nn.Module):
        """Wrap a module whose output is concatenated with its input.

        Args:
            fn: Module applied to the input before channel-wise concatenation.
        """

        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate transformed and original channels.

        Args:
            x: Input tensor passed through ``fn`` and concatenated with the
                result.
        """

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
        use_optimal_head: bool = False,
        use_optimal_backbone: bool = False,
    ):
        """Create the compact LegNet regression model.

        Args:
            in_ch: Number of input channels in encoded sequence tensors.
            stem_ch: Number of channels produced by the stem block.
            stem_ks: Kernel size for the stem convolution.
            ef_ks: Kernel size for efficient and local blocks after the stem.
            ef_block_sizes: Output channel sizes for successive LegNet stages.
            pool_sizes: Pooling factors for successive LegNet stages.
            resize_factor: Expansion factor inside efficient blocks.
            block_dropout: Dropout probability inside convolutional blocks.
            head_dropout: Dropout probability in the regression head.
            stem_dropout: Dropout probability in the stem block.
            output_dim: Number of output units produced by the head.
            use_optimal_head: Whether to use ``Linear -> SiLU -> BatchNorm1d ->
                Dropout -> Linear`` in the head instead of ``Linear ->
                BatchNorm1d -> SiLU -> Dropout -> Linear``.
            use_optimal_backbone: Whether to order stage convolutional blocks as
                ``Conv1d(bias=False) -> [MaxPool1d] -> BatchNorm1d -> SiLU ->
                Dropout1d`` and order the mapper as ``Conv1d(bias=False) ->
                BatchNorm1d -> SiLU``.
        """

        super().__init__()
        if len(pool_sizes) != len(ef_block_sizes):
            raise ValueError("pool_sizes and ef_block_sizes must have the same length")
        self.use_optimal_head = bool(use_optimal_head)
        self.use_optimal_backbone = bool(use_optimal_backbone)
        self.stem = LocalBlock(
            in_ch,
            stem_ch,
            stem_ks,
            dropout=stem_dropout,
            use_optimal_backbone=self.use_optimal_backbone,
        )
        blocks: list[nn.Module] = []
        ch = int(stem_ch)
        for pool, out_ch in zip(pool_sizes, ef_block_sizes):
            blocks.append(
                nn.Sequential(
                    ResidualConcat(EffBlock(ch, ef_ks, resize_factor, dropout=block_dropout)),
                    LocalBlock(
                        ch * 2,
                        int(out_ch),
                        ef_ks,
                        dropout=block_dropout,
                        pool=int(pool) if self.use_optimal_backbone else 1,
                        use_optimal_backbone=self.use_optimal_backbone,
                    ),
                    (
                        nn.Identity()
                        if self.use_optimal_backbone or int(pool) <= 1
                        else nn.MaxPool1d(int(pool))
                    ),
                )
            )
            ch = int(out_ch)
        self.main = nn.Sequential(*blocks)
        if self.use_optimal_backbone:
            self.mapper = nn.Sequential(
                nn.Conv1d(ch, ch * 2, kernel_size=1, bias=False),
                nn.BatchNorm1d(ch * 2),
                nn.SiLU(),
            )
        else:
            self.mapper = nn.Sequential(nn.BatchNorm1d(ch), nn.Conv1d(ch, ch * 2, kernel_size=1))
        if self.use_optimal_head:
            self.head = nn.Sequential(
                nn.Linear(ch * 2, ch * 2),
                nn.SiLU(),
                nn.BatchNorm1d(ch * 2),
                dropout_or_identity(head_dropout),
                nn.Linear(ch * 2, output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(ch * 2, ch * 2),
                nn.BatchNorm1d(ch * 2),
                nn.SiLU(),
                dropout_or_identity(head_dropout),
                nn.Linear(ch * 2, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on ``(B, C, L)`` encoded sequences.

        Args:
            x: Encoded sequence batch with shape ``(batch, channels, length)``.
        """

        z = self.stem(x.float())
        z = self.main(z)
        z = self.mapper(z)
        z = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        return self.head(z).squeeze(-1)
