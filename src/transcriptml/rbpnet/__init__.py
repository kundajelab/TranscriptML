"""RBPNet/eCLIP transcript-space data preparation.

The public API deliberately stops at model-ready data. Neural-network
architectures, losses, and training are not part of this module.
"""

__all__ = [
    "PipelineConfig",
    "ProcessedECLIPDataset",
    "RBPNetBundleConfig",
    "Sample",
    "SelectionConfig",
    "WindowScanConfig",
    "make_rbpnet_bundle",
    "preprocess_eclip",
    "scan_windows",
    "select_regions",
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
    raise AttributeError(f"module 'transcriptml.rbpnet' has no attribute {name!r}")
