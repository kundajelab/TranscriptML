"""Model architectures and registry."""

from transcriptml.models.cnn import SmallCNN, SmallCNNConfig
from transcriptml.models.legnet import LegNet, LegNetConfig
from transcriptml.models.registry import ModelConfig, build_model, load_checkpoint, save_checkpoint
from transcriptml.models.rbpnet import RBPNet, RBPNetConfig, RBPNetOutput
from transcriptml.models.reproduce import SalukiExact, SalukiExactConfig
from transcriptml.models.saluki import SalukiLike, SalukiLikeConfig
from transcriptml.models.saluki_cnn import SalukiCNNHybrid, SalukiCNNHybridConfig
from transcriptml.models.universe import ConvolutionUniverse, ConvolutionUniverseConfig

__all__ = [
    "LegNet",
    "LegNetConfig",
    "ConvolutionUniverse",
    "ConvolutionUniverseConfig",
    "ModelConfig",
    "RBPNet",
    "RBPNetConfig",
    "RBPNetOutput",
    "SalukiExact",
    "SalukiExactConfig",
    "SalukiCNNHybrid",
    "SalukiCNNHybridConfig",
    "SalukiLike",
    "SalukiLikeConfig",
    "SmallCNN",
    "SmallCNNConfig",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
]
