import csv
import json

import numpy as np
import pytest

from transcriptml.data.encoding import encode_saluki_transcript
from transcriptml.data.schemas import SequenceSchema
from transcriptml.interpret.region_ablation import (
    REGION_ABLATION_FAMILIES,
    RegionAblationConfig,
    effective_junction_spacing,
    region_ablation,
    sample_junction_positions,
    save_region_ablation_result,
)


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, X, batch_size=None):
        arr = np.asarray(X)
        self.calls.append(arr.copy())
        base_score = arr[:, 0].sum(axis=1) + 2 * arr[:, 1].sum(axis=1)
        junction_score = 0.25 * arr[:, -1].sum(axis=1)
        return (base_score + junction_score).astype(np.float32)


def _only(*enabled):
    enabled_set = set(enabled)
    return {family: int(family in enabled_set) for family in REGION_ABLATION_FAMILIES}


def _coding_example(length=24):
    return encode_saluki_transcript(
        "ACGUAAACCCGGGUUUCGUA",
        length=length,
        cds_positions=[4, 7, 10, 13],
        splice_positions=[2, 8, 17],
    ).astype(np.float32)


def _decode_codon(x, start):
    symbols = np.argmax(x[:4, start : start + 3], axis=0)
    return "".join("ACGU"[int(symbol)] for symbol in symbols)


def test_region_sequence_families_preserve_annotations_and_target_bounds():
    X = _coding_example()[None]
    predictor = RecordingPredictor()
    config = RegionAblationConfig(
        n_ablations=1,
        n_ablations_for=_only(*(family for family in REGION_ABLATION_FAMILIES if family != "junction_scatter")),
        junction_counts=(1,),
        seed=7,
    )
    result = region_ablation(
        X,
        predictor,
        sequence_ids=["coding"],
        metadata=[{"cds_length": 12}],
        config=config,
        mutation_batch_size=20,
        progress=False,
    )

    assert [row.operation for row in result.instances] == list(REGION_ABLATION_FAMILIES[:-1])
    mutants = predictor.calls[1]
    assert mutants.shape[0] == 8
    for mutant in mutants:
        np.testing.assert_array_equal(mutant[4:], X[0, 4:])

    before = np.argmax(X[0, :4], axis=0)
    bounds = {row.operation: (row.region_start, row.region_end) for row in result.instances}
    for row, mutant in zip(result.instances, mutants):
        start, end = bounds[row.operation]
        if row.operation.endswith("_shuffle") and row.operation != "cds_codon_shuffle":
            after = np.argmax(mutant[:4], axis=0)
            assert sorted(after[start:end].tolist()) == sorted(before[start:end].tolist())
        if row.operation.endswith("_random"):
            assert np.all(mutant[:4, start:end].sum(axis=0) == 1)

    codon_row = next(row for row in result.instances if row.operation == "cds_codon_shuffle")
    codon_mutant = mutants[codon_row.instance_index]
    original_codons = [X[0, :4, i : i + 3].tobytes() for i in range(4, 16, 3)]
    mutant_codons = [codon_mutant[:4, i : i + 3].tobytes() for i in range(4, 16, 3)]
    assert sorted(original_codons) == sorted(mutant_codons)

    synonymous_row = next(row for row in result.instances if row.operation == "cds_synonymous")
    synonymous_mutant = mutants[synonymous_row.instance_index]
    amino_acid_families = (
        {"AAA", "AAG"},
        {"CCU", "CCC", "CCA", "CCG"},
        {"GGU", "GGC", "GGA", "GGG"},
        {"UUU", "UUC"},
    )
    for start, family in zip(range(4, 16, 3), amino_acid_families, strict=True):
        assert _decode_codon(synonymous_mutant, start) in family - {_decode_codon(X[0], start)}


def test_synonymous_cds_ablation_preserves_stops_singletons_and_ambiguous_codons():
    sequence = "AGCUUGGAUGUAAUAGUGANNNC"
    X = encode_saluki_transcript(
        sequence,
        length=len(sequence),
        cds_positions=[1, 4, 7, 10, 13, 16, 19],
        splice_positions=[2, 17],
    )[None].astype(np.float32)
    predictor = RecordingPredictor()
    result = region_ablation(
        X,
        predictor,
        metadata=[{"cds_length": 21}],
        config=RegionAblationConfig(
            n_ablations=12,
            n_ablations_for={
                **_only("cds_synonymous"),
                "cds_synonymous": 12,
            },
            junction_counts=(1,),
            seed=41,
        ),
        mutation_batch_size=20,
        progress=False,
    )

    assert [row.operation for row in result.instances] == ["cds_synonymous"]
    mutants = predictor.calls[1]
    assert mutants.shape == (12, *X.shape[1:])
    for mutant in mutants:
        assert _decode_codon(mutant, 1) in {"GCC", "GCA", "GCG"}
        assert _decode_codon(mutant, 4) == "UGG"
        assert _decode_codon(mutant, 7) == "AUG"
        assert _decode_codon(mutant, 10) == "UAA"
        assert _decode_codon(mutant, 13) == "UAG"
        assert _decode_codon(mutant, 16) == "UGA"
        np.testing.assert_array_equal(mutant[:4, 19:22], X[0, :4, 19:22])
        np.testing.assert_array_equal(mutant[:4][:, [0, 22]], X[0, :4][:, [0, 22]])
        np.testing.assert_array_equal(mutant[4:], X[0, 4:])


def test_junction_scatter_coding_and_noncoding_scope_and_soft_spacing():
    coding = encode_saluki_transcript(
        "A" * 60,
        length=60,
        cds_positions=list(range(5, 56, 3)),
        splice_positions=[2, 10, 30, 57],
    ).astype(np.float32)
    noncoding = encode_saluki_transcript(
        "C" * 60,
        length=60,
        splice_positions=[3, 20, 58],
    ).astype(np.float32)
    X = np.stack([coding, noncoding])
    predictor = RecordingPredictor()
    result = region_ablation(
        X,
        predictor,
        sequence_ids=["coding", "nc"],
        metadata=[{"cds_length": 51}, {"cds_length": 0}],
        config=RegionAblationConfig(
            n_ablations=1,
            n_ablations_for=_only("junction_scatter"),
            junction_counts=(1, 5, 50),
            junction_min_spacing=25,
            seed=19,
        ),
        mutation_batch_size=20,
        progress=False,
    )

    assert [(row.seq_index, row.region, row.junction_count) for row in result.instances] == [
        (0, "cds", 1),
        (0, "cds", 5),
        (0, "cds", 50),
        (1, "transcript", 1),
        (1, "transcript", 5),
        (1, "transcript", 50),
    ]
    mutants = predictor.calls[1]
    for row, mutant in zip(result.instances, mutants):
        marks = np.flatnonzero(mutant[5, row.region_start : row.region_end])
        assert marks.size == row.junction_count
        assert np.all(marks < row.region_length - 1)
        if marks.size > 1:
            assert np.diff(marks).min() >= row.effective_min_spacing
        np.testing.assert_array_equal(mutant[:5], X[row.seq_index, :5])
        if row.seq_index == 0:
            assert mutant[5, 2] == 1
            assert mutant[5, 57] == 1
    assert next(row for row in result.instances if row.seq_index == 0 and row.junction_count == 50).effective_min_spacing == 1


def test_default_grid_has_19_coding_and_11_noncoding_conditions():
    coding = encode_saluki_transcript(
        "A" * 70,
        length=70,
        cds_positions=list(range(5, 62, 3)),
        splice_positions=[10, 30],
    )
    noncoding = encode_saluki_transcript("C" * 70, length=70, splice_positions=[20])
    result = region_ablation(
        np.stack([coding, noncoding]).astype(np.float32),
        RecordingPredictor(),
        config=RegionAblationConfig(n_ablations=1),
        progress=False,
    )

    counts = {seq_index: 0 for seq_index in (0, 1)}
    for row in result.instances:
        counts[row.seq_index] += 1
    assert counts == {0: 19, 1: 11}
    assert int(result.replicate_mask.sum()) == 30


def test_junction_sampling_is_exact_unique_and_rejection_free():
    assert effective_junction_spacing(51, 50, 25) == 1
    assert effective_junction_spacing(100, 5, 25) == 24
    rng = np.random.default_rng(3)
    positions = sample_junction_positions(
        start=7,
        end=107,
        junction_count=5,
        min_spacing=24,
        rng=rng,
    )
    assert positions.shape == (5,)
    assert len(set(positions.tolist())) == 5
    assert positions.min() >= 7 and positions.max() <= 105
    assert np.diff(positions).min() >= 24
    with pytest.raises(ValueError, match="greater than"):
        effective_junction_spacing(5, 5, 25)


def test_unresolved_coding_and_infeasible_noncoding_conditions_are_audited():
    X = np.stack(
        [
            encode_saluki_transcript("A" * 10, length=10),
            encode_saluki_transcript("C" * 3, length=10),
        ]
    ).astype(np.float32)
    result = region_ablation(
        X,
        RecordingPredictor(),
        sequence_ids=["unresolved", "nc"],
        metadata=[{"cds_length": 9}, {"cds_length": 0}],
        config=RegionAblationConfig(
            n_ablations=1,
            n_ablations_for=_only("junction_scatter"),
            junction_counts=(1, 5),
        ),
        progress=False,
    )

    assert [(row.seq_index, row.junction_count) for row in result.instances] == [(1, 1)]
    assert [(row.seq_index, row.junction_count, row.reason) for row in result.skipped] == [
        (0, 1, "represented_cds_missing"),
        (0, 5, "represented_cds_missing"),
        (1, 5, "region_length_not_greater_than_junction_count"),
    ]
    assert result.transcript_classes == ("coding_unresolved", "noncoding")


def test_region_ablation_reproducible_across_batching_and_sharding():
    X = np.stack([_coding_example(), _coding_example()])
    config = RegionAblationConfig(
        n_ablations=3,
        n_ablations_for=_only("cds_random", "cds_synonymous", "junction_scatter"),
        junction_counts=(1, 5),
        seed=29,
    )
    full = region_ablation(
        X,
        RecordingPredictor(),
        sequence_ids=["a", "b"],
        config=config,
        mutation_batch_size=2,
        progress=False,
    )
    shard = region_ablation(
        X,
        RecordingPredictor(),
        sequence_ids=["a", "b"],
        config=config,
        mutation_batch_size=17,
        sequence_shard_index=1,
        sequence_shards=2,
        progress=False,
    )

    full_rows = [row for row in full.instances if row.seq_index == 1]
    assert [(row.operation, row.junction_count) for row in full_rows] == [
        (row.operation, row.junction_count) for row in shard.instances
    ]
    full_indices = [row.instance_index for row in full_rows]
    np.testing.assert_array_equal(full.effects[full_indices], shard.effects)


def test_storage_serialization_and_custom_channels(tmp_path):
    X = _coding_example()[None]
    schema = SequenceSchema(
        name="custom_saluki",
        channels=("A", "C", "G", "U", "coding_marks", "junction_marks"),
    )
    out = tmp_path / "region"
    result = region_ablation(
        X,
        RecordingPredictor(),
        schema=schema,
        sequence_ids=["tx1"],
        config=RegionAblationConfig(
            n_ablations=2,
            n_ablations_for={"junction_scatter": 1},
            junction_counts=(1,),
        ),
        cds_channel="coding_marks",
        splice_channel="junction_marks",
        storage_dir=out,
        progress=False,
    )
    save_region_ablation_result(
        result,
        out,
        checkpoint="fold0/model/best.pt",
        dataset="data/saluki",
        progress=False,
    )

    expected = {
        "reference_predictions.npy",
        "ablation_predictions.npy",
        "effects.npy",
        "replicate_mask.npy",
        "mean_effects.npy",
        "mean_abs_effects.npy",
        "std_effects.npy",
        "instances.csv",
        "skipped.csv",
        "summary.json",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis"] == "region_ablation"
    assert summary["n_mutants"] == int(result.replicate_mask.sum())
    assert summary["cds_channel_index"] == 4
    assert summary["splice_channel_index"] == 5
    with (out / "instances.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sequence_id"] == "tx1"
    np.testing.assert_array_equal(np.load(out / "replicate_mask.npy"), result.replicate_mask)


def test_region_ablation_config_validation():
    with pytest.raises(ValueError, match="unique"):
        RegionAblationConfig(junction_counts=(1, 1)).normalized()
    with pytest.raises(ValueError, match="Unknown"):
        RegionAblationConfig(n_ablations_for={"bad": 1}).normalized()
    with pytest.raises(ValueError, match="non-negative"):
        RegionAblationConfig(n_ablations_for={"cds_random": -1}).normalized()
