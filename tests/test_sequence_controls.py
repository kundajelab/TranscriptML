import numpy as np
import pytest

from transcriptml.data.bundle import DatasetBundle, load_bundle
from transcriptml.data.controls import (
    apply_sequence_controls_array,
    apply_sequence_controls_to_bundle,
    normalize_sequence_control_config,
)
from transcriptml.data.encoding import decode_rna_one_hot, encode_saluki_transcript


def _example_saluki_batch() -> np.ndarray:
    seq = "ACGUAAACCCGGGUUUCGUA"
    x = encode_saluki_transcript(
        seq,
        length=24,
        cds_positions=[4, 7, 10, 13],
        splice_positions=[3, 15],
    )
    return x[None].astype(np.uint8)


def _decode_valid(x: np.ndarray) -> str:
    return decode_rna_one_hot(x[:4])[:20]


def test_shuffle_nucleotides_targets_requested_regions_only():
    X = _example_saluki_batch()
    out, stats = apply_sequence_controls_array(
        X,
        {"shuffle_nucleotides": ["5utr", "3utr"], "seed": 7},
        progress=False,
    )

    before = _decode_valid(X[0])
    after = _decode_valid(out[0])
    assert sorted(after[:4]) == sorted(before[:4])
    assert after[4:16] == before[4:16]
    assert sorted(after[16:20]) == sorted(before[16:20])
    np.testing.assert_array_equal(out[0, 4:], X[0, 4:])
    assert stats["edited"]["shuffle_nucleotides"]["5utr"] == 1
    assert stats["edited"]["shuffle_nucleotides"]["3utr"] == 1


def test_shuffle_codons_permutes_cds_codon_units_and_preserves_annotations():
    X = _example_saluki_batch()
    out, stats = apply_sequence_controls_array(X, {"shuffle_codons": True, "seed": 3}, progress=False)

    before = _decode_valid(X[0])
    after = _decode_valid(out[0])
    before_codons = [before[i : i + 3] for i in range(4, 16, 3)]
    after_codons = [after[i : i + 3] for i in range(4, 16, 3)]
    assert after[:4] == before[:4]
    assert after[16:20] == before[16:20]
    assert sorted(after_codons) == sorted(before_codons)
    np.testing.assert_array_equal(out[0, 4:], X[0, 4:])
    assert stats["edited"]["shuffle_codons"]["cds"] == 1


def test_randomize_nucleotides_preserves_region_length_annotations_and_padding():
    X = _example_saluki_batch()
    out, stats = apply_sequence_controls_array(
        X,
        {"randomize_nucleotides": ["3utr"], "seed": 11},
        progress=False,
    )

    before = _decode_valid(X[0])
    after = _decode_valid(out[0])
    assert after[:16] == before[:16]
    assert len(after[16:20]) == len(before[16:20])
    assert np.all(out[0, :4, 16:20].sum(axis=0) == 1)
    assert np.all(out[0, :, 20:] == 0)
    np.testing.assert_array_equal(out[0, 4:], X[0, 4:])
    assert stats["edited"]["randomize_nucleotides"]["3utr"] == 1


def test_cds_frameshift_shifts_cds_channel_only():
    X = _example_saluki_batch()
    out, stats = apply_sequence_controls_array(X, {"cds_frameshift": 1}, progress=False)

    np.testing.assert_array_equal(out[0, :4], X[0, :4])
    np.testing.assert_array_equal(out[0, 5], X[0, 5])
    assert np.flatnonzero(X[0, 4]).tolist() == [4, 7, 10, 13]
    assert np.flatnonzero(out[0, 4]).tolist() == [5, 8, 11, 14]
    assert stats["edited"]["cds_frameshift"]["cds"] == 1


def test_cds_frameshift_explicit_operation_accepts_shift_two():
    X = _example_saluki_batch()
    out, _ = apply_sequence_controls_array(
        X,
        {"operations": [{"operation": "cds_frameshift", "shift": 2}]},
        progress=False,
    )

    assert np.flatnonzero(out[0, 4]).tolist() == [6, 9, 12, 15]


def test_cds_frameshift_rejects_invalid_shift():
    with pytest.raises(ValueError):
        normalize_sequence_control_config({"cds_frameshift": 3})


def test_legacy_ablation_names_normalize_to_explicit_operations():
    cfg = normalize_sequence_control_config(
        {
            "5pUTR_ablation": "scramble",
            "CDS_ablation": "scramble",
            "3pUTR_ablation": "ablate",
            "ablation_seed": 99,
            "seed": 99,
        }
    )
    by_operation = {op.operation: op.regions for op in cfg.operations}
    assert by_operation["shuffle_nucleotides"] == ("5utr",)
    assert by_operation["shuffle_codons"] == ("cds",)
    assert by_operation["randomize_nucleotides"] == ("3utr",)
    assert cfg.seed == 99


def test_sequence_controls_can_save_controlled_bundle(tmp_path):
    X = _example_saluki_batch()
    bundle = DatasetBundle(
        X=X,
        y=np.array([1.25], dtype=np.float32),
        ids=["tx1"],
        schema="saluki6",
        metadata=[{"gene_id": "g1"}],
        splits={"train": [0], "val": [], "test": []},
        config={"builder": "test"},
    )
    out_dir = tmp_path / "controlled"
    controlled, stats = apply_sequence_controls_to_bundle(
        bundle,
        {"randomize_nucleotides": ["5utr"], "seed": 5, "save_dir": str(out_dir)},
        progress=False,
    )

    assert stats["save_dir"] == str(out_dir)
    assert (out_dir / "X.npy").exists()
    assert (out_dir / "sequence_controls.json").exists()
    loaded = load_bundle(out_dir)
    np.testing.assert_array_equal(loaded.X, controlled.X)
    np.testing.assert_allclose(loaded.y, bundle.y)
    assert loaded.ids == ["tx1"]
    assert loaded.config["sequence_controls"]["operations"][0]["operation"] == "randomize_nucleotides"
