"""Model architectures and registry."""

from transcriptml.models.cnn import SmallCNN, SmallCNNConfig
from transcriptml.models.legnet import LegNet, LegNetConfig
from transcriptml.models.registry import ModelConfig, build_model, load_checkpoint, save_checkpoint
from transcriptml.models.reproduce import SalukiExact, SalukiExactConfig
from transcriptml.models.saluki import SalukiLike, SalukiLikeConfig

__all__ = [
    "LegNet",
    "LegNetConfig",
    "ModelConfig",
    "SalukiExact",
    "SalukiExactConfig",
    "SalukiLike",
    "SalukiLikeConfig",
    "SmallCNN",
    "SmallCNNConfig",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
]
