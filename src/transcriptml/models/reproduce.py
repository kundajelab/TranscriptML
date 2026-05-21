from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from transcriptml.models.common import ChannelLayerNorm


@dataclass
class SalukiExactConfig:
    seq_depth: int = 6
    filters: int = 64
    kernel_size: int = 5
    num_layers: int = 6
    dropout: float = 0.3
    augment_shift: int = 3
    ln_epsilon: float = 0.007
    keras_bn_momentum: float = 0.90
    bn_eps: float = 1e-3

    def to_kwargs(self) -> dict[str, object]:
        return asdict(self)


def keras_he_normal_trunc_(tensor: torch.Tensor) -> torch.Tensor:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / fan_in)
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def shift_sequence_right(x: torch.Tensor, shift: int, pad_value: float = 0.0) -> torch.Tensor:
    if shift == 0:
        return x
    if shift < 0:
        raise ValueError("Only non-negative shifts are supported.")
    pad = x.new_full((x.shape[0], x.shape[1], shift), pad_value)
    return torch.cat([pad, x[:, :, :-shift]], dim=-1)


class StochasticShift(nn.Module):
    def __init__(self, shift_max: int = 3):
        super().__init__()
        self.shift_max = int(shift_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.shift_max <= 0:
            return x
        shift = int(torch.randint(0, self.shift_max + 1, size=(), device=x.device).item())
        return shift_sequence_right(x, shift)


class SalukiExact(nn.Module):
    """Close PyTorch reproduction of the Basenji/Saluki rnann.py architecture."""

    def __init__(
        self,
        seq_depth: int = 6,
        filters: int = 64,
        kernel_size: int = 5,
        num_layers: int = 6,
        dropout: float = 0.3,
        augment_shift: int = 3,
        ln_epsilon: float = 0.007,
        keras_bn_momentum: float = 0.90,
        bn_eps: float = 1e-3,
    ):
        super().__init__()
        self.seq_depth = int(seq_depth)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.num_layers = int(num_layers)
        bn_momentum_pt = 1.0 - float(keras_bn_momentum)
        self.shift = StochasticShift(augment_shift)
        self.conv0 = nn.Conv1d(seq_depth, filters, kernel_size=kernel_size, padding=0, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln": ChannelLayerNorm(filters, eps=ln_epsilon),
                        "act": nn.ReLU(),
                        "conv": nn.Conv1d(filters, filters, kernel_size=kernel_size, padding=0),
                        "drop": nn.Dropout(dropout),
                        "pool": nn.MaxPool1d(kernel_size=2, stride=2),
                    }
                )
            )
        self.pre_rnn_ln = ChannelLayerNorm(filters, eps=ln_epsilon)
        self.pre_rnn_act = nn.ReLU()
        self.gru = nn.GRU(input_size=filters, hidden_size=filters, batch_first=True)
        self.bn1 = nn.BatchNorm1d(filters, eps=bn_eps, momentum=bn_momentum_pt)
        self.fc1 = nn.Linear(filters, filters)
        self.drop1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(filters, eps=bn_eps, momentum=bn_momentum_pt)
        self.fc2 = nn.Linear(filters, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        keras_he_normal_trunc_(self.conv0.weight)
        for blk in self.blocks:
            keras_he_normal_trunc_(blk["conv"].weight)
            nn.init.zeros_(blk["conv"].bias)
        keras_he_normal_trunc_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)
        keras_he_normal_trunc_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        keras_he_normal_trunc_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got {tuple(x.shape)}")
        if x.shape[1] == self.seq_depth:
            return x.float()
        if x.shape[-1] == self.seq_depth:
            return x.transpose(1, 2).float()
        raise ValueError(f"Expected channel depth {self.seq_depth}, got shape {tuple(x.shape)}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.shift(self._to_bcl(x))
        z = self.conv0(z)
        for blk in self.blocks:
            z = blk["ln"](z)
            z = blk["act"](z)
            z = blk["conv"](z)
            z = blk["drop"](z)
            z = blk["pool"](z)
        z = self.pre_rnn_act(self.pre_rnn_ln(z)).transpose(1, 2)
        z = torch.flip(z, dims=[1])
        z, _ = self.gru(z)
        return z[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        z = self.fc1(torch.relu(self.bn1(z)))
        z = self.drop1(z)
        return self.fc2(torch.relu(self.bn2(z))).squeeze(-1)
