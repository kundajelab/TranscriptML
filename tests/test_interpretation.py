import numpy as np
import torch

from transcriptml.data.encoding import encode_rna_sequence
from transcriptml.interpret.ablation import motif_ablation
from transcriptml.interpret.context import motif_context_scan
from transcriptml.interpret.epistasis import motif_epistasis
from transcriptml.interpret.ism import compute_ism
from transcriptml.interpret.predictor import Predictor


class BaseWeightModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).view(1, 4, 1))

    def forward(self, x):
        return (x[:, :4, :] * self.weights).sum(dim=(1, 2))


class InteractionModel(torch.nn.Module):
    def forward(self, x):
        return x[:, 0, 0] * x[:, 0, 1] * x[:, 0, 2] * x[:, 0, 3]


def test_ism_known_signed_effects():
    X = encode_rna_sequence("AC")[None, :, :].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))
    result = compute_ism(X, predictor, mutation_batch_size=2)
    assert result.reference_predictions.tolist() == [3.0]
    assert result.deltas[0, 1, 0] == 1.0
    assert result.deltas[0, 2, 0] == 3.0
    assert result.deltas[0, 3, 0] == 7.0
    assert result.deltas[0, 0, 1] == -1.0
    assert result.deltas[0, 2, 1] == 2.0
    assert result.deltas[0, 3, 1] == 6.0


def test_motif_ablation_known_additive_effect():
    X = encode_rna_sequence("AAUU")[None, :, :].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 0, 0, 0]))
    result = motif_ablation(X, predictor, motif="AA", n_scrambles=3, seed=1)
    assert len(result.instances) == 1
    assert result.reference_predictions[0] == 2.0
    assert result.ablation_predictions[0] == 0.0
    assert result.effects[0] == -2.0


def test_context_scan_additive_model_cancels_to_zero():
    X = encode_rna_sequence("AAUUAA")[None, :, :].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 0, 0, 0]))
    result = motif_context_scan(
        X,
        predictor,
        motif="AA",
        window_size=2,
        context_width=4,
        n_motif_scrambles=2,
        n_window_scrambles=2,
        seed=2,
    )
    assert result.context_mask.sum() > 0
    np.testing.assert_allclose(result.context_effects[result.context_mask.astype(bool)], 0.0, atol=1e-6)


def test_epistasis_known_interaction_term():
    X = encode_rna_sequence("AAAA")[None, :, :].astype(np.float32)
    predictor = Predictor(InteractionModel())
    result = motif_epistasis(X, predictor, motif="AA", n_scrambles=3, seed=3)
    assert len(result.pairs) == 1
    assert result.reference_predictions[0] == 1.0
    assert result.single_ablation_predictions[0, 0] == 0.0
    assert result.single_ablation_predictions[0, 1] == 0.0
    assert result.paired_ablation_predictions[0] == 0.0
    assert result.epistasis[0] == 1.0
