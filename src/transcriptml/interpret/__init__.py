"""Interpretation analyses."""

from __future__ import annotations

import importlib

__all__ = [
    "EnsemblePredictor",
    "DeepSHAPAttributionResult",
    "GeneratedReferences",
    "LegNetScanResult",
    "Predictor",
    "ReferenceConfig",
    "ReferenceDiagnostic",
    "ReferenceEvaluation",
    "ReferenceGenerator",
    "compute_codon_ism",
    "compute_deep_shap_attributions",
    "compute_ism",
    "compute_window_ism",
    "motif_ablation",
    "motif_context_scan",
    "motif_epistasis",
    "evaluate_references",
    "region_ablation",
    "run_deep_shap",
    "save_deep_shap_result",
    "scan_legnet_windows",
    "slice_attribution_inputs",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "motif_ablation": ("transcriptml.interpret.ablation", "motif_ablation"),
    "motif_context_scan": ("transcriptml.interpret.context", "motif_context_scan"),
    "compute_codon_ism": ("transcriptml.interpret.codon_ism", "compute_codon_ism"),
    "DeepSHAPAttributionResult": ("transcriptml.interpret.deep_shap", "DeepSHAPAttributionResult"),
    "GeneratedReferences": ("transcriptml.interpret.deep_shap", "GeneratedReferences"),
    "ReferenceConfig": ("transcriptml.interpret.deep_shap", "ReferenceConfig"),
    "ReferenceDiagnostic": ("transcriptml.interpret.deep_shap", "ReferenceDiagnostic"),
    "ReferenceEvaluation": ("transcriptml.interpret.deep_shap", "ReferenceEvaluation"),
    "ReferenceGenerator": ("transcriptml.interpret.deep_shap", "ReferenceGenerator"),
    "compute_deep_shap_attributions": ("transcriptml.interpret.deep_shap", "compute_deep_shap_attributions"),
    "evaluate_references": ("transcriptml.interpret.deep_shap", "evaluate_references"),
    "run_deep_shap": ("transcriptml.interpret.deep_shap", "run_deep_shap"),
    "save_deep_shap_result": ("transcriptml.interpret.deep_shap", "save_deep_shap_result"),
    "slice_attribution_inputs": ("transcriptml.interpret.deep_shap", "slice_attribution_inputs"),
    "motif_epistasis": ("transcriptml.interpret.epistasis", "motif_epistasis"),
    "compute_ism": ("transcriptml.interpret.ism", "compute_ism"),
    "LegNetScanResult": ("transcriptml.interpret.legnet_scan", "LegNetScanResult"),
    "scan_legnet_windows": ("transcriptml.interpret.legnet_scan", "scan_legnet_windows"),
    "EnsemblePredictor": ("transcriptml.interpret.predictor", "EnsemblePredictor"),
    "Predictor": ("transcriptml.interpret.predictor", "Predictor"),
    "region_ablation": ("transcriptml.interpret.region_ablation", "region_ablation"),
    "compute_window_ism": ("transcriptml.interpret.window_ism", "compute_window_ism"),
}


def __getattr__(name: str):
    """Lazily load interpretation APIs to avoid eager heavy imports at CLI startup."""

    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'transcriptml.interpret' has no attribute {name!r}")
    module_name, attr_name = spec
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
