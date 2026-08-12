"""RBPNet/eCLIP preprocessing, structured data, losses, and training APIs."""

__all__ = [
    "PipelineConfig",
    "ProcessedECLIPDataset",
    "RBPNetBatch",
    "RBPNetBundleConfig",
    "RBPNetDataset",
    "RBPNetLossConfig",
    "RBPNetObjective",
    "Sample",
    "SelectionConfig",
    "WindowScanConfig",
    "make_rbpnet_bundle",
    "evaluate_rbpnet_model",
    "preprocess_eclip",
    "scan_windows",
    "select_regions",
    "train_rbpnet_model",
    "write_rbpnet_predictions",
]


def __getattr__(name: str):
    """Lazily expose APIs so base TranscriptML installs can still show CLI help."""

    if name in {"PipelineConfig", "Sample", "preprocess_eclip"}:
        from transcriptml.rbpnet.preprocessing import PipelineConfig, Sample, preprocess_eclip

        return {"PipelineConfig": PipelineConfig, "Sample": Sample, "preprocess_eclip": preprocess_eclip}[name]
    if name == "ProcessedECLIPDataset":
        from transcriptml.rbpnet.experiment import ProcessedECLIPDataset

        return ProcessedECLIPDataset
    if name in {"WindowScanConfig", "scan_windows"}:
        from transcriptml.rbpnet.windows import WindowScanConfig, scan_windows

        return {"WindowScanConfig": WindowScanConfig, "scan_windows": scan_windows}[name]
    if name in {"SelectionConfig", "select_regions"}:
        from transcriptml.rbpnet.selection import SelectionConfig, select_regions

        return {"SelectionConfig": SelectionConfig, "select_regions": select_regions}[name]
    if name in {"RBPNetBundleConfig", "make_rbpnet_bundle"}:
        from transcriptml.rbpnet.bundle import RBPNetBundleConfig, make_rbpnet_bundle

        return {"RBPNetBundleConfig": RBPNetBundleConfig, "make_rbpnet_bundle": make_rbpnet_bundle}[name]
    if name in {"RBPNetBatch", "RBPNetDataset"}:
        from transcriptml.rbpnet.dataset import RBPNetBatch, RBPNetDataset

        return {"RBPNetBatch": RBPNetBatch, "RBPNetDataset": RBPNetDataset}[name]
    if name in {"RBPNetLossConfig", "RBPNetObjective"}:
        from transcriptml.rbpnet.losses import RBPNetLossConfig, RBPNetObjective

        return {"RBPNetLossConfig": RBPNetLossConfig, "RBPNetObjective": RBPNetObjective}[name]
    if name in {"evaluate_rbpnet_model", "train_rbpnet_model", "write_rbpnet_predictions"}:
        from transcriptml.rbpnet.training import (
            evaluate_rbpnet_model,
            train_rbpnet_model,
            write_rbpnet_predictions,
        )

        return {
            "evaluate_rbpnet_model": evaluate_rbpnet_model,
            "train_rbpnet_model": train_rbpnet_model,
            "write_rbpnet_predictions": write_rbpnet_predictions,
        }[name]
    raise AttributeError(f"module 'transcriptml.rbpnet' has no attribute {name!r}")
