from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    """Resolve ``auto`` or explicit torch device names to a torch device."""

    if isinstance(device, torch.device):
        return device
    name = str(device)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
