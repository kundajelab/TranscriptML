import json

import numpy as np
import pytest
import torch

from transcriptml.data.encoding import encode_rna_sequence, encode_saluki_transcript
from transcriptml.interpret.ablation import motif_ablation
from transcriptml.interpret.context import motif_context_scan
from transcriptml.interpret.codon_ism import NpzMutationTableWriter, compute_codon_ism, find_cds_codon_starts
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


def test_motif_ablation_region_filter_skips_nonmatching_regions():
    X = encode_saluki_transcript("AAACCCGGGAA", length=11, cds_positions=[3, 6])[None].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 0, 0, 0]))

    utr5 = motif_ablation(X, predictor, motif="AA", n_scrambles=0, region="5utr", progress=False)
    utr3 = motif_ablation(X, predictor, motif="AA", n_scrambles=0, region="3'UTR", progress=False)
    cds = motif_ablation(X, predictor, motif="AA", n_scrambles=0, region="CDS", progress=False)

    assert [(inst.start, inst.end, inst.region) for inst in utr5.instances] == [(0, 2, "5utr"), (1, 3, "5utr")]
    assert [(inst.start, inst.end, inst.region) for inst in utr3.instances] == [(9, 11, "3utr")]
    assert cds.instances == []


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


def test_codon_ism_synonymous_table_and_position_scores():
    X = encode_saluki_transcript("AUGGCUUGG", length=9, cds_positions=[0, 3, 6])[None].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))
    result = compute_codon_ism(
        X,
        predictor,
        mutation_batch_size=2,
        compute_position_scores=True,
    )

    rows = result.mutations
    assert rows.shape[0] == 3
    assert rows["sequence_index"].tolist() == [0, 0, 0]
    assert rows["codon_start"].tolist() == [3, 3, 3]
    assert rows["reference_codon"].tolist() == ["GCU", "GCU", "GCU"]
    assert rows["alternate_codon"].tolist() == ["GCC", "GCA", "GCG"]
    assert rows["reference_amino_acid"].tolist() == ["A", "A", "A"]
    assert rows["alternate_amino_acid"].tolist() == ["A", "A", "A"]
    assert rows["synonymous"].tolist() == [True, True, True]
    np.testing.assert_allclose(rows["reference_prediction"], np.array([43.0, 43.0, 43.0]))
    np.testing.assert_allclose(rows["delta"], np.array([-6.0, -7.0, -4.0]))
    np.testing.assert_allclose(rows["cds_relative_position"], np.array([5 / 9, 5 / 9, 5 / 9]))

    assert result.position_scores.shape == (1, 4, 9)
    assert result.position_scores[0, 2, 3] == 7.0
    assert result.position_scores[0, 1, 4] == 7.0
    assert result.position_scores[0, 3, 5] == 7.0
    assert np.count_nonzero(result.position_scores) == 3


def test_codon_ism_all_codons_stop_filter_and_dense_cds():
    X = encode_saluki_transcript("AUG", length=3, cds_positions=[0, 1, 2])[None].astype(np.float32)
    cds = find_cds_codon_starts(X[0], "saluki6", valid_length=3)
    assert cds.encoding == "dense"
    assert cds.starts.tolist() == [0]

    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))
    with_stops = compute_codon_ism(
        X,
        predictor,
        mutation_policy="all-codons",
        mutation_batch_size=128,
    )
    without_stops = compute_codon_ism(
        X,
        predictor,
        mutation_policy="all-codons",
        include_stop_codons=False,
        mutation_batch_size=128,
    )

    assert with_stops.mutations.shape[0] == 63
    assert without_stops.mutations.shape[0] == 60
    assert {"UAA", "UAG", "UGA"}.issubset(set(with_stops.mutations["alternate_codon"].tolist()))
    assert not {"UAA", "UAG", "UGA"} & set(without_stops.mutations["alternate_codon"].tolist())


def test_codon_ism_streams_to_npz_without_collecting(tmp_path):
    X = encode_saluki_transcript("GCU", length=3, cds_positions=[0])[None].astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))
    writer = NpzMutationTableWriter(tmp_path / "mutations_npz", rows_per_shard=2)
    result = compute_codon_ism(
        X,
        predictor,
        mutation_batch_size=1,
        writer=writer,
        collect=False,
    )

    assert result.mutations.shape[0] == 0
    manifest = json.loads((tmp_path / "mutations_npz" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_rows"] == 3
    assert [part["n_rows"] for part in manifest["parts"]] == [2, 1]
    first = np.load(tmp_path / "mutations_npz" / manifest["parts"][0]["path"])
    assert first["reference_codon"].tolist() == ["GCU", "GCU"]


def test_codon_ism_sequence_slice_preserves_original_indices():
    X = np.stack(
        [
            encode_saluki_transcript("GCU", length=3, cds_positions=[0]),
            encode_saluki_transcript("GCC", length=3, cds_positions=[0]),
            encode_saluki_transcript("GCA", length=3, cds_positions=[0]),
        ]
    ).astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))

    result = compute_codon_ism(
        X,
        predictor,
        sequence_start=1,
        sequence_end=3,
        mutation_batch_size=2,
        progress=False,
    )

    assert result.sequence_indices.tolist() == [1, 2]
    assert result.reference_predictions.shape == (2,)
    assert sorted(set(result.mutations["sequence_index"].tolist())) == [1, 2]


def test_codon_ism_sequence_shards_are_contiguous_and_exclusive():
    X = np.stack(
        [encode_saluki_transcript("GCU", length=3, cds_positions=[0]) for _ in range(7)]
    ).astype(np.float32)
    predictor = Predictor(BaseWeightModel([1, 2, 4, 8]))

    middle = compute_codon_ism(
        X,
        predictor,
        sequence_shard_index=1,
        sequence_shards=3,
        mutation_batch_size=8,
        progress=False,
    )
    last = compute_codon_ism(
        X,
        predictor,
        sequence_shard_index=2,
        sequence_shards=3,
        mutation_batch_size=8,
        progress=False,
    )

    assert middle.sequence_indices.tolist() == [2, 3]
    assert last.sequence_indices.tolist() == [4, 5, 6]
    with pytest.raises(ValueError, match="Pass only one"):
        compute_codon_ism(
            X,
            predictor,
            sequence_start=0,
            sequence_shard_index=0,
            sequence_shards=3,
            progress=False,
        )
