"""Interpretation analyses."""

from transcriptml.interpret.ablation import motif_ablation
from transcriptml.interpret.context import motif_context_scan
from transcriptml.interpret.codon_ism import compute_codon_ism
from transcriptml.interpret.deep_shap import (
    DeepSHAPAttributionResult,
    GeneratedReferences,
    ReferenceConfig,
    ReferenceDiagnostic,
    ReferenceEvaluation,
    ReferenceGenerator,
    compute_deep_shap_attributions,
    evaluate_references,
    run_deep_shap,
    save_deep_shap_result,
    slice_attribution_inputs,
)
from transcriptml.interpret.epistasis import motif_epistasis
from transcriptml.interpret.ism import compute_ism
from transcriptml.interpret.legnet_scan import LegNetScanResult, scan_legnet_windows
from transcriptml.interpret.predictor import EnsemblePredictor, Predictor
from transcriptml.interpret.region_ablation import region_ablation
from transcriptml.interpret.window_ism import compute_window_ism

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
