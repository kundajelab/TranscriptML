from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

from transcriptml.cli.main import build_parser, main
from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.interpret.deep_shap import (
    GeneratedReferences,
    ReferenceConfig,
    ReferenceDiagnostic,
    ReferenceEvaluation,
    ReferenceGenerator,
    _RecurrentSummary,
    _TargetPredictor,
    _adapt_attribution_model,
    compute_deep_shap_attributions,
    evaluate_references,
)
from transcriptml.interpret.predictor import Predictor
from transcriptml.models.legnet import LegNet
from transcriptml.models.reproduce import SalukiExact
from transcriptml.models.registry import save_checkpoint
from transcriptml.models.saluki import SalukiLike
from transcriptml.models.saluki_cnn import SalukiCNNHybrid
from transcriptml.models.universe import ConvolutionUniverse


BASES = "ACGU"


def _saluki_sequence(
    sequence: str, codon_starts: tuple[int, ...], *, length: int = 24
) -> np.ndarray:
    X = np.zeros((6, length), dtype=np.int8)
    for position, base in enumerate(sequence):
        X[BASES.index(base), position] = 1
    X[4, list(codon_starts)] = 1
    X[5, [4, 10]] = 1
    return X


def _decode(X: np.ndarray) -> str:
    return "".join(BASES[int(index)] for index in X[:4].argmax(axis=0))


def _dinucleotides(sequence: str) -> Counter[str]:
    return Counter(sequence[index : index + 2] for index in range(len(sequence) - 1))


@pytest.fixture
def coding_sequence() -> np.ndarray:
    # 5' UTR AC; CDS AUG/GCU/GAA/UAA; 3' UTR CGGC; remaining columns are padding.
    return _saluki_sequence("ACAUGGCUGAAUAACGGC", (2, 5, 8, 11))


def test_reference_generation_preserves_regions_annotations_and_padding(coding_sequence):
    mono = ReferenceGenerator(
        ReferenceConfig(strategy="5utr-mono", n_references=6, seed=4), schema="saluki6"
    ).generate(coding_sequence, sequence_index=9)
    assert mono.eligible
    assert mono.references is not None
    for reference in mono.references:
        np.testing.assert_array_equal(reference[:, 2:], coding_sequence[:, 2:])
        np.testing.assert_array_equal(reference[4:], coding_sequence[4:])
        assert sorted(_decode(reference[:, :2])) == sorted(_decode(coding_sequence[:, :2]))

    full = ReferenceGenerator(
        ReferenceConfig(strategy="full-mono", n_references=3, seed=5), schema="saluki6"
    ).generate(coding_sequence, sequence_index=9)
    assert full.references is not None
    valid_length = 18
    for reference in full.references:
        np.testing.assert_array_equal(reference[4:], coding_sequence[4:])
        np.testing.assert_array_equal(reference[:4, valid_length:], 0)
        np.testing.assert_array_equal(
            reference[:4, :valid_length].sum(axis=1),
            coding_sequence[:4, :valid_length].sum(axis=1),
        )


def test_dinucleotide_reference_preserves_region_dinucleotides(coding_sequence):
    generated = ReferenceGenerator(
        ReferenceConfig(strategy="3utr-dinuc", n_references=4, seed=12), schema="saluki6"
    ).generate(coding_sequence, sequence_index=2)
    assert generated.references is not None
    original_region = _decode(coding_sequence[:, 14:18])
    for reference in generated.references:
        np.testing.assert_array_equal(reference[:, :14], coding_sequence[:, :14])
        np.testing.assert_array_equal(reference[:, 18:], coding_sequence[:, 18:])
        assert _dinucleotides(_decode(reference[:, 14:18])) == _dinucleotides(original_region)


def test_codon_reference_strategies_preserve_required_properties(coding_sequence):
    codons = ("AUG", "GCU", "GAA", "UAA")
    for strategy in ("cds-codon-shuffle", "cds-synonymous", "cds-unrestricted"):
        generator = ReferenceGenerator(
            ReferenceConfig(strategy=strategy, n_references=8, seed=17), schema="saluki6"
        )
        generated = generator.generate(coding_sequence, sequence_index=3)
        assert generated.references is not None
        for reference in generated.references:
            decoded = _decode(reference[:, 2:14])
            observed = tuple(decoded[index : index + 3] for index in range(0, 12, 3))
            assert observed[0] == "AUG"
            assert observed[-1] == "UAA"
            np.testing.assert_array_equal(reference[:, :2], coding_sequence[:, :2])
            np.testing.assert_array_equal(reference[:, 14:], coding_sequence[:, 14:])
            np.testing.assert_array_equal(reference[4:], coding_sequence[4:])
            if strategy == "cds-codon-shuffle":
                assert Counter(observed) == Counter(codons)
            elif strategy == "cds-synonymous":
                assert observed[1] in {"GCU", "GCC", "GCA", "GCG"}
                assert observed[2] in {"GAA", "GAG"}
            else:
                assert observed[1] not in {"UAA", "UAG", "UGA"}
                assert observed[2] not in {"UAA", "UAG", "UGA"}

    stop_recoding = ReferenceGenerator(
        ReferenceConfig(
            strategy="cds-synonymous",
            n_references=4,
            seed=17,
            preserve_terminal_stop_codon=False,
            include_stop_codons=True,
        ),
        schema="saluki6",
    ).generate(coding_sequence, sequence_index=3)
    assert stop_recoding.references is not None
    for reference in stop_recoding.references:
        assert _decode(reference[:, 11:14]) in {"UAG", "UGA"}


def test_references_are_statelessly_reproducible(coding_sequence):
    generator = ReferenceGenerator(
        ReferenceConfig(strategy="cds-unrestricted", n_references=5, seed=91), schema="saluki6"
    )
    first = generator.generate(coding_sequence, sequence_index=22)
    generator.generate(coding_sequence, sequence_index=1)
    second = generator.generate(coding_sequence, sequence_index=22)
    np.testing.assert_array_equal(first.references, second.references)


def test_missing_cds_and_empty_region_are_ineligible(coding_sequence):
    missing = coding_sequence.copy()
    missing[4] = 0
    generated = ReferenceGenerator(
        ReferenceConfig(strategy="cds-synonymous"), schema="saluki6"
    ).generate(missing, sequence_index=0)
    assert not generated.eligible
    assert generated.reason == "missing_cds"
    assert generated.references is None

    no_utr = _saluki_sequence("AUGGCUUAA", (0, 3, 6), length=12)
    generated = ReferenceGenerator(
        ReferenceConfig(strategy="5utr-mono"), schema="saluki6"
    ).generate(no_utr, sequence_index=0)
    assert not generated.eligible
    assert generated.reason == "empty_5utr"


class _FixedGenerator:
    def __init__(self, references):
        self.references = references
        self.config = ReferenceConfig(n_references=len(references))

    def generate(self, x, *, sequence_index):
        return GeneratedReferences(self.references.copy(), True, None, 1, 0, 1)


def test_reference_diagnostics_confidence_ties_and_effect_size(tmp_path: Path):
    X = np.asarray([[[[10.0]]], [[[10.0]]]]).reshape(2, 1, 1)
    references = np.asarray([[[12.0]], [[11.0]], [[10.0]], [[13.0]]])
    evaluation = evaluate_references(
        X,
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(references),
        sequence_ids=("a", "b"),
        confidence_threshold=0.75,
        minimum_effect_size=1.0,
        zero_atol=1e-6,
        reference_predictions_path=tmp_path / "references.csv",
        reference_predictions_format="csv",
    )
    row = evaluation.diagnostics[0]
    assert row.reference_prediction_mean == pytest.approx(11.5)
    assert row.reference_prediction_sd == pytest.approx(np.std([12, 11, 10, 13]))
    assert row.mean_reference_minus_original == pytest.approx(1.5)
    assert row.mean_absolute_prediction_difference == pytest.approx(1.5)
    assert row.fraction_above == 0.75
    assert row.fraction_below == 0.0
    assert row.fraction_near_zero == 0.25
    assert row.maximum_directional_consistency == 0.75
    assert row.confidence_classification == "high_confidence"
    assert len((tmp_path / "references.csv").read_text().splitlines()) == 9

    ties = evaluate_references(
        X[:1],
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(np.full((4, 1, 1), 10.0)),
        confidence_threshold=0.95,
    )
    assert ties.diagnostics[0].fraction_near_zero == 1.0
    assert ties.diagnostics[0].confidence_classification == "low_confidence"

    too_small = evaluate_references(
        X[:1],
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(np.full((4, 1, 1), 10.5)),
        confidence_threshold=0.95,
        minimum_effect_size=1.0,
    )
    assert too_small.diagnostics[0].maximum_directional_consistency == 1.0
    assert too_small.diagnostics[0].confidence_classification == "low_confidence"

    empty = evaluate_references(
        X,
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(references),
        sequence_indices=(),
    )
    assert empty.diagnostics == []
    assert empty.original_predictions.shape == (0,)


def test_reference_predictions_parquet(tmp_path: Path):
    parquet = pytest.importorskip("pyarrow.parquet")
    X = np.full((1, 1, 1), 10.0)
    references = np.asarray([[[9.0]], [[11.0]], [[12.0]]])
    path = tmp_path / "references.parquet"
    evaluate_references(
        X,
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(references),
        reference_predictions_path=path,
        reference_predictions_format="parquet",
    )
    table = parquet.read_table(path)
    assert table.num_rows == 3
    assert table.column_names == [
        "sequence_index",
        "sequence_id",
        "strategy",
        "reference_index",
        "prediction",
        "difference_from_original",
        "direction",
    ]


def test_reference_evaluation_progress_reports_confidence_counts(capsys):
    X = np.full((2, 1, 1), 10.0)
    references = np.asarray([[[11.0]], [[12.0]]])
    evaluate_references(
        X,
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(references),
        confidence_threshold=1.0,
    )

    stderr = capsys.readouterr().err
    assert "deep-shap: predicting original outputs for 2 sequences" in stderr
    assert "deep-shap: reference evaluation: started (0/2 sequences)" in stderr
    assert "finished (2/2 sequences); high=2, low=0, ineligible=0" in stderr


def test_reference_evaluation_progress_can_be_disabled(capsys):
    X = np.full((1, 1, 1), 10.0)
    references = np.asarray([[[11.0]]])
    evaluate_references(
        X,
        lambda batch: np.asarray(batch)[:, 0, 0],
        _FixedGenerator(references),
        progress=False,
    )

    assert capsys.readouterr().err == ""


def _random_ohe(n: int, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 4, size=(n, length))
    return np.eye(4, dtype=np.float32)[indices].transpose(0, 2, 1)


def test_saluki_and_legnet_attribution_adapters_preserve_predictions():
    torch.manual_seed(3)
    saluki = SalukiLike(
        in_ch=6,
        base_ch=4,
        n_convs=1,
        pool_size=2,
        dropout=0,
        gru_hidden=4,
        head_hidden=4,
    ).eval()
    X = np.concatenate((_random_ohe(2, 16), np.zeros((2, 2, 16), dtype=np.float32)), axis=1)
    with torch.no_grad():
        expected = saluki(torch.tensor(X))
        observed = _adapt_attribution_model(saluki)(torch.tensor(X))
    torch.testing.assert_close(observed, expected)

    legnet = LegNet(
        stem_ch=4,
        stem_ks=3,
        ef_ks=3,
        ef_block_sizes=(4,),
        pool_sizes=(2,),
        resize_factor=2,
        head_dropout=0,
        output_dim=2,
    ).eval()
    X4 = torch.tensor(_random_ohe(2, 16, seed=4))
    with torch.no_grad():
        torch.testing.assert_close(_adapt_attribution_model(legnet)(X4), legnet(X4))
    target_predictions = _TargetPredictor(
        Predictor(legnet, device="cpu", batch_size=2), target=1, batch_size=2
    ).predict(X4)
    with torch.no_grad():
        np.testing.assert_allclose(target_predictions, legnet(X4)[:, 1].numpy(), atol=1e-6)


def test_other_recurrent_attribution_adapters_preserve_predictions():
    torch.manual_seed(31)
    X = torch.tensor(
        np.concatenate(
            (_random_ohe(2, 20, seed=32), np.zeros((2, 2, 20), dtype=np.float32)),
            axis=1,
        )
    )
    models = [
        SalukiExact(
            filters=3,
            kernel_size=3,
            num_layers=1,
            dropout=0,
            augment_shift=0,
        ),
        SalukiCNNHybrid(
            saluki={
                "seq_depth": 6,
                "filters": 3,
                "kernel_size": 3,
                "num_layers": 1,
                "dropout": 0,
                "augment_shift": 0,
            },
            cnn={
                "in_ch": 6,
                "n_filters": 3,
                "kernel_size": 3,
                "n_layers": 1,
                "dropout": 0,
            },
            fusion={
                "saluki_projection_dim": 3,
                "cnn_projection_dim": 3,
                "head_hidden": 3,
                "projection_dropout": 0,
                "head_dropout": 0,
            },
        ),
        ConvolutionUniverse(
            in_ch=6,
            filters=3,
            num_layers=1,
            stem_kernel_size=3,
            kernel_size=3,
            dilations=1,
            use_residual=False,
            pooling="max",
            pool_size=2,
            dropout=0,
            rnn_type="gru",
            rnn_hidden=3,
            rnn_summary="last",
            head_hidden=3,
            head_dropout=0,
        ),
    ]
    with torch.no_grad():
        for model in models:
            model.eval()
            torch.testing.assert_close(_adapt_attribution_model(model)(X), model(X))


def test_saluki_exact_adapter_exposes_functional_head_relu_modules():
    model = SalukiExact(
        filters=3,
        kernel_size=3,
        num_layers=1,
        dropout=0,
        augment_shift=0,
    ).eval()
    adapted = _adapt_attribution_model(model)

    assert isinstance(adapted.head_act1, torch.nn.ReLU)
    assert isinstance(adapted.head_act2, torch.nn.ReLU)
    X = torch.tensor(
        np.concatenate(
            (_random_ohe(2, 20, seed=41), np.zeros((2, 2, 20), dtype=np.float32)),
            axis=1,
        )
    )
    with torch.no_grad():
        torch.testing.assert_close(adapted(X), model(X))


def test_tiny_saluki_deep_shap_converges_and_preserves_actual_relation():
    torch.manual_seed(8)
    model = SalukiLike(
        in_ch=6,
        base_ch=3,
        n_convs=1,
        pool_size=2,
        dropout=0,
        gru_hidden=3,
        head_hidden=3,
    ).eval()
    bases = _random_ohe(1, 12, seed=9).astype(np.int8)
    bases[:, :, 9:] = 0
    X = np.concatenate((bases, np.zeros((1, 2, 12), dtype=np.int8)), axis=1)
    predictor = Predictor(model, device="cpu", batch_size=4)
    generator = ReferenceGenerator(
        ReferenceConfig(strategy="full-mono", n_references=2, seed=5), schema="saluki6"
    )
    evaluation = evaluate_references(
        X,
        _TargetPredictor(predictor, target=0, batch_size=4),
        generator,
        confidence_threshold=0.0,
    )
    result = compute_deep_shap_attributions(
        X,
        predictor,
        generator,
        evaluation,
        attribution_batch_size=1,
        ig_points=8,
        max_ig_points=16,
        convergence_atol=2e-3,
        convergence_rtol=0.02,
    )
    hypothetical = result.high_confidence_hypothetical_attributions
    actual = result.high_confidence_actual_attributions
    np.testing.assert_allclose(actual, hypothetical * X[:, :4], atol=1e-6)
    np.testing.assert_array_equal(hypothetical[:, :, 9:], 0)
    assert evaluation.diagnostics[0].convergence_absolute_error is not None
    assert evaluation.diagnostics[0].convergence_pass is True
    assert evaluation.diagnostics[0].ig_points_used in {8, 16}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cuDNN RNN")
def test_tiny_saluki_deep_shap_supports_cudnn_gru_backward():
    """Tangermeme's eval mode must not prevent cuDNN GRU backward."""

    torch.manual_seed(18)
    model = SalukiLike(
        in_ch=6,
        base_ch=2,
        n_convs=1,
        pool_size=2,
        dropout=0,
        gru_hidden=2,
        head_hidden=2,
    ).eval()
    bases = _random_ohe(1, 10, seed=19).astype(np.int8)
    X = np.concatenate((bases, np.zeros((1, 2, 10), dtype=np.int8)), axis=1)
    predictor = Predictor(model, device="cuda", batch_size=2)
    generator = ReferenceGenerator(
        ReferenceConfig(strategy="full-mono", n_references=1, seed=7),
        schema="saluki6",
    )
    evaluation = evaluate_references(
        X,
        _TargetPredictor(predictor, target=0, batch_size=2),
        generator,
        confidence_threshold=0.0,
        progress=False,
    )
    assert model.gru.training is False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = compute_deep_shap_attributions(
            X,
            predictor,
            generator,
            evaluation,
            attribution_batch_size=1,
            ig_points=2,
            max_ig_points=2,
            convergence_atol=1.0,
            convergence_rtol=1.0,
            progress=False,
        )

    assert result.high_confidence_actual_attributions.shape == (1, 4, 10)
    assert model.training is False
    assert model.gru.training is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cuDNN RNN")
def test_cuda_recurrent_attribution_keeps_stacked_dropout_disabled():
    """The cuDNN workaround must preserve eval semantics for stacked RNNs."""

    torch.manual_seed(20)
    recurrent = torch.nn.GRU(
        input_size=3,
        hidden_size=4,
        num_layers=2,
        batch_first=True,
        dropout=0.5,
    ).cuda().eval()
    summary = _RecurrentSummary(
        recurrent,
        summary="state-last",
        layers=2,
    ).cuda().eval()
    X = torch.randn(2, 5, 3, device="cuda")
    with torch.no_grad():
        expected = summary(X)

    grad_input = X.detach().clone().requires_grad_()
    observed = summary(grad_input)
    observed.sum().backward()

    # Native and cuDNN recurrent kernels differ slightly in floating-point
    # accumulation order, while remaining numerically equivalent.
    torch.testing.assert_close(observed, expected, atol=1e-4, rtol=5e-4)
    assert grad_input.grad is not None
    assert recurrent.training is False


def _controlled_attribution_case():
    model = SalukiLike(
        in_ch=6,
        base_ch=2,
        n_convs=1,
        pool_size=2,
        dropout=0,
        gru_hidden=2,
        head_hidden=2,
    ).eval()
    X = np.concatenate(
        (_random_ohe(1, 8, seed=21), np.zeros((1, 2, 8), dtype=np.float32)), axis=1
    )
    generator = ReferenceGenerator(
        ReferenceConfig(n_references=1, seed=3), schema="saluki6"
    )
    diagnostic = ReferenceDiagnostic(
        sequence_index=0,
        sequence_id="0",
        strategy="full-mono",
        eligible=True,
        eligibility_reason=None,
        n_references=1,
        original_prediction=1.0,
        reference_prediction_mean=0.0,
        confidence_classification="high_confidence",
        valid_length=8,
    )
    evaluation = ReferenceEvaluation(
        sequence_indices=np.asarray([0]),
        diagnostics=[diagnostic],
        original_predictions=np.asarray([1.0], dtype=np.float32),
        high_confidence_indices=np.asarray([0]),
        low_confidence_indices=np.empty(0, dtype=np.int64),
        ineligible_indices=np.empty(0, dtype=np.int64),
        confidence_threshold=0.95,
        minimum_effect_size=None,
        zero_atol=1e-6,
    )
    return X, Predictor(model, device="cpu"), generator, evaluation


def test_local_ig_retry_escalation(monkeypatch, capsys):
    import transcriptml.interpret.deep_shap as deep_shap_module

    X, predictor, generator, evaluation = _controlled_attribution_case()
    calls = []

    def fake_deep_lift(_model, sequences, **kwargs):
        calls.append(kwargs["additional_nonlinear_ops"])
        if len(calls) < 3:
            return torch.zeros_like(sequences)
        return torch.full_like(sequences, 1.0 / sequences.shape[-1])

    monkeypatch.setattr(deep_shap_module, "deep_lift_shap", fake_deep_lift)
    result = compute_deep_shap_attributions(
        X,
        predictor,
        generator,
        evaluation,
        ig_points=8,
        max_ig_points=32,
        convergence_atol=1e-6,
        convergence_rtol=1e-6,
    )
    assert len(calls) == 3
    assert evaluation.diagnostics[0].ig_points_used == 32
    assert evaluation.diagnostics[0].convergence_pass is True
    assert result.unresolved_convergence_indices.size == 0
    stderr = capsys.readouterr().err
    assert "deep-shap: attribution: started (0/1 sequences)" in stderr
    assert "retries=2, convergence_failures=0, last_K=32" in stderr
    assert "finished (1/1 sequences); high=1, low=0, retries=2" in stderr


def test_unresolved_convergence_warning_is_persisted(monkeypatch):
    import transcriptml.interpret.deep_shap as deep_shap_module

    X, predictor, generator, evaluation = _controlled_attribution_case()

    def fake_deep_lift(_model, sequences, **_kwargs):
        warnings.warn("Convergence deltas too high: controlled", RuntimeWarning)
        return torch.zeros_like(sequences)

    monkeypatch.setattr(deep_shap_module, "deep_lift_shap", fake_deep_lift)
    with pytest.warns(RuntimeWarning, match="remained outside tolerance"):
        result = compute_deep_shap_attributions(
            X,
            predictor,
            generator,
            evaluation,
            ig_points=8,
            max_ig_points=32,
            convergence_atol=1e-6,
            convergence_rtol=1e-6,
        )
    row = evaluation.diagnostics[0]
    assert row.ig_points_used == 32
    assert row.convergence_pass is False
    assert row.convergence_warning_count == 3
    assert row.convergence_warnings == (
        "K=8: Convergence deltas too high: controlled | "
        "K=16: Convergence deltas too high: controlled | "
        "K=32: Convergence deltas too high: controlled"
    )
    np.testing.assert_array_equal(result.high_confidence_actual_attributions, 0)


def test_run_outputs_and_slice_alignment(tmp_path: Path, capsys):
    torch.manual_seed(11)
    X = _random_ohe(2, 12, seed=13)
    bundle_dir = tmp_path / "bundle"
    save_bundle(
        DatasetBundle(
            X=X,
            ids=("first", "second"),
            schema="rna4",
            metadata=({"group": "a"}, {"group": "b"}),
        ),
        bundle_dir,
    )
    model = LegNet(
        stem_ch=4,
        stem_ks=3,
        ef_ks=3,
        ef_block_sizes=(4,),
        pool_sizes=(2,),
        resize_factor=2,
        head_dropout=0,
    ).eval()
    checkpoint = tmp_path / "legnet.pt"
    save_checkpoint(
        checkpoint,
        model,
        {
            "name": "legnet",
            "params": {
                "stem_ch": 4,
                "stem_ks": 3,
                "ef_ks": 3,
                "ef_block_sizes": [4],
                "pool_sizes": [2],
                "resize_factor": 2,
                "head_dropout": 0,
            },
        },
    )
    out = tmp_path / "deep_shap"
    main(
        [
            "deep-shap",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(bundle_dir),
            "--out-dir",
            str(out),
            "--n-references",
            "2",
            "--seed",
            "2",
            "--confidence-threshold",
            "0",
            "--ig-points",
            "4",
            "--max-ig-points",
            "4",
        ]
    )
    assert (out / "reference_diagnostics.csv").exists()
    assert (out / "summary.json").exists()
    assert len(np.load(out / "high_confidence_indices.npy")) == 2
    assert not list(out.glob(".*.partial.npy"))
    assert json.loads((out / "summary.json").read_text())["convergence"][
        "unresolved_sequence_indices"
    ] == []
    stderr = capsys.readouterr().err
    assert f"deep-shap: saving outputs to {out}" in stderr
    assert "deep-shap: completed; high=2, low=0, ineligible=0" in stderr

    sliced = tmp_path / "sliced"
    main(
        [
            "slice-attribution-inputs",
            "--dataset",
            str(bundle_dir),
            "--attribution-dir",
            str(out),
            "--out-dir",
            str(sliced),
        ]
    )
    sequences = np.load(sliced / "sequences.npy")
    indices = np.load(sliced / "sequence_indices.npy")
    np.testing.assert_array_equal(sequences, X[indices])
    assert (sliced / "ids.txt").read_text().splitlines() == ["first", "second"]


def test_deep_shap_cli_defaults():
    args = build_parser().parse_args(
        ["deep-shap", "--checkpoint", "model.pt", "--dataset", "data", "--out-dir", "out"]
    )
    assert args.reference_strategy == "full-mono"
    assert args.n_references == 20
    assert args.seed == 123
    assert args.target == 0

    slice_args = build_parser().parse_args(
        [
            "slice-attribution-inputs",
            "--dataset",
            "data",
            "--attribution-dir",
            "attrs",
            "--out-dir",
            "sliced",
        ]
    )
    assert slice_args.all_channels is True
