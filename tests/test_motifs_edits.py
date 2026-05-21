import numpy as np
import pytest

from transcriptml.data.encoding import encode_rna_sequence
from transcriptml.interpret.edits import scramble_motif_ablating_inplace
from transcriptml.interpret.motifs import find_motif_starts, parse_motif


def test_motif_parsing_and_degenerate_scan():
    motif = parse_motif("[A|G|U]GAC[U|A|C]")
    assert motif[0] == {0, 2, 3}
    assert motif[-1] == {0, 1, 3}

    x = encode_rna_sequence("UGACU")
    assert find_motif_starts(x, "[A|G|U]GAC[U|A|C]").tolist() == [0]


def test_all_zero_positions_do_not_match_wildcards():
    x = encode_rna_sequence("ANG")
    assert find_motif_starts(x, "A.G").tolist() == []


def test_scrambling_preserves_annotation_channels():
    x = np.zeros((6, 5), dtype=np.float32)
    x[:4] = encode_rna_sequence("AAUU", length=5)
    x[4:, :] = np.arange(10, dtype=np.float32).reshape(2, 5)
    annotations = x[4:].copy()
    scramble_motif_ablating_inplace(
        x,
        motif_start=0,
        motif_sets=parse_motif("AA"),
        strategy="random_different",
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(x[4:], annotations)
    assert find_motif_starts(x[:4, :2], "AA").tolist() == []


def test_scrambling_rejects_all_zero_region():
    x = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        scramble_motif_ablating_inplace(
            x,
            motif_start=0,
            motif_sets=parse_motif("A"),
            strategy="random_different",
            rng=np.random.default_rng(1),
        )
