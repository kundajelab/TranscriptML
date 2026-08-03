import json

import numpy as np
import pytest
import torch

from transcriptml.analysis.window_ism_summary import summarize_window_ism_folds
from transcriptml.cli.main import main
from transcriptml.data.encoding import encode_rna_sequence, encode_saluki_transcript, encode_sequences
from transcriptml.interpret.predictor import Predictor
from transcriptml.interpret.window_ism import (
    WindowISMResult,
    compute_window_ism,
    generate_window_starts,
    save_window_ism_result,
)


class BaseWeightModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).view(1, 4, 1))

    def forward(self, x):
        return (x[:, :4, :] * self.weights).sum(dim=(1, 2))


class PositionAModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).view(1, 1, -1))

    def forward(self, x):
        return (x[:, 0:1, :] * self.weights).sum(dim=(1, 2))


class RecordingCallable:
    def __init__(self):
        self.calls = []

    def __call__(self, X):
        values = np.asarray(X)
        self.calls.append(values.copy())
        return values[:, :4, :].sum(axis=(1, 2), dtype=np.float32)


def test_generate_window_starts_anchors_terminal_window_and_covers_sequence():
    starts = generate_window_starts(valid_length=10, window_size=4, stride=4)
    assert starts.tolist() == [0, 4, 6]
    covered = np.zeros(10, dtype=bool)
    for start in starts:
        covered[start : start + 4] = True
    assert covered.all()

    assert generate_window_starts(valid_length=3, window_size=4, stride=4).size == 0
    with pytest.raises(ValueError, match="stride"):
        generate_window_starts(valid_length=10, window_size=4, stride=5)


@pytest.mark.parametrize(("length", "stride"), [(4, 4), (5, 4), (10, 4), (10, 3), (11, 1)])
def test_scored_windows_cover_every_unambiguous_base(length, stride):
    X = encode_rna_sequence("A" * length)[None].astype(np.float32)
    result = compute_window_ism(
        X,
        Predictor(BaseWeightModel([1, 0, 0, 0])),
        window_size=4,
        stride=stride,
        n_ablations=1,
        progress=False,
    )
    covered = np.zeros(length, dtype=bool)
    for window_i in np.flatnonzero(result.window_mask[0]):
        start = int(result.window_starts[0, window_i])
        covered[start : start + result.window_size] = True
    assert covered.all()


def test_window_ism_exact_additive_effects_and_terminal_coordinates():
    X = encode_rna_sequence("AAAAA")[None].astype(np.float32)
    result = compute_window_ism(
        X,
        Predictor(BaseWeightModel([1, 0, 0, 0])),
        window_size=2,
        stride=2,
        n_ablations=3,
        mutation_batch_size=2,
        progress=False,
    )

    assert result.window_starts.tolist() == [[0, 2, 3]]
    assert result.window_mask.tolist() == [[True, True, True]]
    np.testing.assert_allclose(result.mean_deltas, -2.0)
    np.testing.assert_allclose(result.mean_abs_deltas, 2.0)
    np.testing.assert_allclose(result.std_deltas, 0.0)
    assert result.reference_predictions.tolist() == [5.0]


def test_window_ism_compact_padding_ambiguous_mask_and_short_sequence():
    X = encode_sequences(["ACGUACGUAA", "ACGUACGU", "ACN"], length=10).astype(np.float32)
    result = compute_window_ism(
        X,
        Predictor(BaseWeightModel([1, 2, 4, 8])),
        window_size=4,
        stride=4,
        n_ablations=2,
        progress=False,
    )

    assert result.window_starts.tolist() == [[0, 4, 6], [0, 4, -1], [-1, -1, -1]]
    assert result.window_mask.tolist() == [[True, True, True], [True, True, False], [False, False, False]]
    assert np.all(result.mean_deltas[~result.window_mask] == 0)

    ambiguous = encode_rna_sequence("ACNU")[None].astype(np.float32)
    ambiguous_result = compute_window_ism(
        ambiguous,
        Predictor(BaseWeightModel([1, 2, 4, 8])),
        window_size=2,
        stride=2,
        n_ablations=2,
        progress=False,
    )
    assert ambiguous_result.window_starts.tolist() == [[0, 2]]
    assert ambiguous_result.window_mask.tolist() == [[True, False]]


def test_window_ism_preserves_annotation_channels():
    X = encode_saluki_transcript(
        "ACGUAC",
        length=6,
        cds_positions=[0, 3],
        splice_positions=[2, 5],
    )[None].astype(np.float32)
    recorder = RecordingCallable()
    compute_window_ism(
        X,
        Predictor(recorder),
        window_size=3,
        stride=3,
        n_ablations=2,
        mutation_batch_size=3,
        progress=False,
    )

    mutant_calls = recorder.calls[1:]
    assert mutant_calls
    for call in mutant_calls:
        expected_annotations = np.repeat(X[:, 4:, :], call.shape[0], axis=0)
        np.testing.assert_array_equal(call[:, 4:, :], expected_annotations)
        reference_bases = np.argmax(X[0, :4, :], axis=0)
        for mutant in call:
            mutant_bases = np.argmax(mutant[:4, :], axis=0)
            assert np.count_nonzero(mutant_bases != reference_bases) == 3


def test_window_ism_is_reproducible_across_mutation_batch_sizes_and_seeded():
    X = encode_rna_sequence("AAAAAA")[None].astype(np.float32)
    predictor = Predictor(BaseWeightModel([0, 1, 4, 9]))
    kwargs = dict(window_size=3, stride=2, n_ablations=12, seed=17, progress=False)
    first = compute_window_ism(X, predictor, mutation_batch_size=1, **kwargs)
    second = compute_window_ism(X, predictor, mutation_batch_size=7, **kwargs)
    np.testing.assert_array_equal(first.mean_deltas, second.mean_deltas)
    np.testing.assert_array_equal(first.mean_abs_deltas, second.mean_abs_deltas)
    np.testing.assert_array_equal(first.std_deltas, second.std_deltas)

    different = compute_window_ism(X, predictor, mutation_batch_size=7, **{**kwargs, "seed": 18})
    assert not np.array_equal(first.mean_deltas, different.mean_deltas)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_size": 0}, "window_size"),
        ({"window_size": 3, "stride": 4}, "stride"),
        ({"window_size": 3, "n_ablations": 0}, "n_ablations"),
        ({"window_size": 3, "mutation_batch_size": 0}, "mutation_batch_size"),
        ({"window_size": 3, "seed": -1}, "seed"),
    ],
)
def test_window_ism_rejects_invalid_parameters(kwargs, message):
    X = encode_rna_sequence("AAAA")[None].astype(np.float32)
    with pytest.raises(ValueError, match=message):
        compute_window_ism(X, Predictor(BaseWeightModel([1, 0, 0, 0])), progress=False, **kwargs)


def test_window_ism_mean_absolute_effect_ranks_high_weight_window():
    X = encode_rna_sequence("AAAAAA")[None].astype(np.float32)
    predictor = Predictor(PositionAModel([1, 1, 10, 10, 1, 1]))
    result = compute_window_ism(
        X,
        predictor,
        window_size=2,
        stride=2,
        n_ablations=3,
        progress=False,
    )
    assert int(np.argmax(result.mean_abs_deltas[0])) == 1
    np.testing.assert_allclose(result.mean_abs_deltas[0], [2, 20, 2])


def _synthetic_result(mean, magnitude, deviation, reference, *, starts=None):
    mean = np.asarray(mean, dtype=np.float32)
    starts_array = np.asarray(starts if starts is not None else [[0, 2]], dtype=np.int64)
    return WindowISMResult(
        window_starts=starts_array,
        window_mask=starts_array >= 0,
        mean_deltas=mean,
        mean_abs_deltas=np.asarray(magnitude, dtype=np.float32),
        std_deltas=np.asarray(deviation, dtype=np.float32),
        reference_predictions=np.asarray(reference, dtype=np.float32),
        valid_lengths=np.asarray([4], dtype=np.int64),
        input_shape=(1, 4, 4),
        window_size=2,
        stride=2,
        n_ablations=30,
        seed=123,
    )


def test_save_window_ism_result_writes_arrays_metadata_and_coverage(tmp_path):
    X = encode_rna_sequence("AAAAA")[None].astype(np.float32)
    result = compute_window_ism(
        X,
        Predictor(BaseWeightModel([1, 0, 0, 0])),
        window_size=2,
        n_ablations=2,
        progress=False,
    )
    save_window_ism_result(
        result,
        tmp_path,
        checkpoint="model.pt",
        dataset="data/bundle",
        sequence_ids=["seq0"],
        progress=False,
    )

    for name in (
        "window_starts",
        "window_mask",
        "mean_deltas",
        "mean_abs_deltas",
        "std_deltas",
        "reference_predictions",
        "valid_lengths",
    ):
        assert (tmp_path / f"{name}.npy").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis"] == "window_ism"
    assert summary["n_valid_bases"] == 5
    assert summary["n_covered_valid_bases"] == 5
    assert summary["n_uncovered_valid_bases"] == 0
    assert summary["sequence_ids_sha256"] is not None


def test_summarize_window_ism_separates_within_and_between_model_variation(tmp_path):
    input_dir = tmp_path / "window_ism"
    fold0 = input_dir / "fold0"
    fold1 = input_dir / "fold1"
    save_window_ism_result(
        _synthetic_result([[1, 3]], [[1, 3]], [[2, 4]], [10]),
        fold0,
        sequence_ids=["seq0"],
        progress=False,
    )
    save_window_ism_result(
        _synthetic_result([[3, 7]], [[3, 7]], [[4, 8]], [14]),
        fold1,
        sequence_ids=["seq0"],
        progress=False,
    )

    out_dir = tmp_path / "summary"
    summary = summarize_window_ism_folds(input_dir=input_dir, out_dir=out_dir, batch_size=1)
    np.testing.assert_allclose(np.load(out_dir / "average_mean_deltas.npy"), [[2, 5]])
    np.testing.assert_allclose(np.load(out_dir / "average_mean_abs_deltas.npy"), [[2, 5]])
    np.testing.assert_allclose(np.load(out_dir / "within_model_std_deltas.npy"), [[np.sqrt(10), np.sqrt(40)]])
    np.testing.assert_allclose(np.load(out_dir / "fold_std_mean_deltas.npy"), [[1, 2]])
    np.testing.assert_allclose(np.load(out_dir / "average_reference_predictions.npy"), [12])
    assert summary["fold_count"] == 2


def test_summarize_window_ism_rejects_coordinate_mismatch(tmp_path):
    input_dir = tmp_path / "window_ism"
    save_window_ism_result(
        _synthetic_result([[1, 2]], [[1, 2]], [[0, 0]], [1]),
        input_dir / "fold0",
        progress=False,
    )
    save_window_ism_result(
        _synthetic_result([[1, 2]], [[1, 2]], [[0, 0]], [1], starts=[[0, 1]]),
        input_dir / "fold1",
        progress=False,
    )
    with pytest.raises(ValueError, match="Coordinate or mask mismatch"):
        summarize_window_ism_folds(input_dir=input_dir, out_dir=tmp_path / "summary")


def test_summarize_window_ism_cli_smoke(tmp_path):
    input_dir = tmp_path / "window_ism"
    save_window_ism_result(
        _synthetic_result([[1, 2]], [[1, 2]], [[0, 0]], [1]),
        input_dir / "fold0",
        progress=False,
    )
    out_dir = tmp_path / "summary"
    main(["summarize-window-ism", "--input-dir", str(input_dir), "--out-dir", str(out_dir)])
    assert (out_dir / "average_mean_deltas.npy").exists()
