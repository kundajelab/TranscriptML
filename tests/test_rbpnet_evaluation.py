import math

import numpy as np
import pytest

from transcriptml.rbpnet.evaluation import resolve_rbpnet_checkpoint_indices
from transcriptml.rbpnet.evaluation_metrics import (
    aggregate_observations,
    calibration_rows,
    enrichment_metrics,
    profile_metrics,
    replicate_ceiling_metrics,
    select_representative_examples,
)


def test_profile_metrics_hand_computable_uniform_control_and_saturated():
    result = profile_metrics(
        np.asarray([[1, 0], [1, 1]], dtype=float),
        np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=float),
        control_probabilities=np.asarray([[1.0, 0.0], [0.5, 0.5]]),
    )
    assert result["multinomial_nll"][0] == pytest.approx(math.log(2))
    assert result["saturated_nll"][0] == pytest.approx(0.0)
    assert result["uniform_nll"][0] == pytest.approx(math.log(2))
    assert result["kl_per_read"][0] == pytest.approx(math.log(2))
    assert result["jsd"][0] == pytest.approx(0.75 * math.log(4 / 3))
    assert result["information_gain_uniform_per_read"][0] == pytest.approx(0.0)
    assert result["information_gain_control_per_read"][0] == pytest.approx(
        -math.log(2)
    )
    assert result["wasserstein_nt"][0] == pytest.approx(0.5)

    # The empirical [0.5, 0.5] profile is saturated by the model. Its complete
    # multinomial NLL is still ln(2), the negative log probability of counts
    # [1,1] under n=2 and p=[0.5,0.5].
    assert result["multinomial_nll"][1] == pytest.approx(math.log(2))
    assert result["multinomial_nll_without_constant"][1] == pytest.approx(
        2 * math.log(2)
    )
    assert result["saturated_nll"][1] == pytest.approx(math.log(2))
    assert result["kl_per_read"][1] == pytest.approx(0.0)


def test_profile_metrics_validity_mask_and_zero_count_behavior():
    result = profile_metrics(
        np.asarray([[0, 1, 0], [0, 0, 0]]),
        np.asarray([[0.25, 0.25, 0.5], [0.1, 0.4, 0.5]]),
        valid_mask=np.asarray([[False, True, True], [False, True, True]]),
    )
    assert result["valid_positions"].tolist() == [2, 2]
    assert result["multinomial_nll"][0] == pytest.approx(math.log(3))
    assert np.isnan(result["kl_per_read"][1])
    assert np.isnan(result["jsd"][1])
    with pytest.raises(ValueError, match="outside the validity mask"):
        profile_metrics(
            np.asarray([[1, 0]]),
            np.asarray([[0.5, 0.5]]),
            valid_mask=np.asarray([[False, True]]),
        )


def test_enrichment_depth_null_and_stabilized_empirical_eta():
    result = enrichment_metrics(
        eta=np.asarray([0.0, math.log(2)]),
        ip_counts=np.asarray([[1], [2]]),
        sminput_counts=np.asarray([1, 0]),
        depth_offsets=np.asarray([0.0]),
        pseudocount=0.5,
    )
    assert result["information_gain_depth_null_per_read"][0, 0] == pytest.approx(0.0)
    assert result["binomial_nll_without_constant"][0, 0] == pytest.approx(
        2 * math.log(2)
    )
    assert result["predicted_probability"][1, 0] == pytest.approx(2 / 3)
    assert result["binomial_nll"][1, 0] == pytest.approx(-2 * math.log(2 / 3))
    assert result["depth_null_nll"][1, 0] == pytest.approx(2 * math.log(2))
    assert result["information_gain_depth_null_per_read"][1, 0] == pytest.approx(
        math.log(4 / 3)
    )
    assert result["empirical_eta"][1, 0] == pytest.approx(math.log(5))
    empty = enrichment_metrics(
        eta=np.asarray([2.0]),
        ip_counts=np.asarray([[0]]),
        sminput_counts=np.asarray([0]),
        depth_offsets=np.asarray([0.0]),
    )
    assert np.isnan(empty["binomial_nll"][0, 0])
    assert np.isnan(empty["information_gain_depth_null_per_read"][0, 0])
    assert np.isnan(empty["empirical_eta"][0, 0])


def test_jsd_wasserstein_and_replicate_ceiling():
    ceiling = replicate_ceiling_metrics(
        np.asarray(
            [
                [[1, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [0, 1, 0]],
            ]
        )
    )
    np.testing.assert_allclose(ceiling["jsd"][0], 0.0)
    np.testing.assert_allclose(ceiling["wasserstein_nt"][0], 0.0)
    np.testing.assert_allclose(ceiling["jsd"][1], math.log(2))
    np.testing.assert_allclose(ceiling["wasserstein_nt"][1], 1.0)
    single = replicate_ceiling_metrics(np.ones((2, 1, 3)))
    assert single["jsd"].shape == (2, 0)
    one_empty = replicate_ceiling_metrics(
        np.asarray([[[1, 0, 0], [0, 0, 0]]])
    )
    assert np.isnan(one_empty["jsd"]).all()
    assert np.isnan(one_empty["wasserstein_nt"]).all()


def test_locus_gene_and_read_aggregation():
    result = aggregate_observations(
        np.asarray([1.0, 3.0, 5.0, 9.0]),
        locus_ids=["a", "a", "b", "c"],
        gene_ids=["g1", "g1", "g1", "g2"],
        read_weights=np.asarray([1.0, 1.0, 2.0, 6.0]),
    )
    assert result["locus_macro"]["value"] == pytest.approx((2 + 5 + 9) / 3)
    assert result["gene_macro"]["value"] == pytest.approx((3.5 + 9) / 2)
    assert result["read_micro"]["value"] == pytest.approx(6.8)
    explicit = aggregate_observations(
        np.asarray([2.0, 4.0]),
        locus_ids=["a", "b"],
        gene_ids=["g1", "g2"],
        read_weights=np.asarray([2.0, 3.0]),
        micro_numerators=np.asarray([2.0, 4.0]),
    )
    assert explicit["read_micro"]["value"] == pytest.approx(6 / 5)


def test_read_weighted_calibration():
    rows = calibration_rows(
        predicted_probability=np.asarray([[0.1], [0.2], [0.9]]),
        ip_counts=np.asarray([[1], [3], [4]]),
        total_counts=np.asarray([[2], [6], [4]]),
        replicate_names=["ip1"],
        n_bins=2,
    )
    assert len(rows) == 2
    low = rows[0]
    assert low["n_observations"] == 2
    assert low["total_reads"] == 8
    assert low["predicted_bin"] == pytest.approx((2 * 0.1 + 6 * 0.2) / 8)
    assert low["observed_bin"] == pytest.approx(4 / 8)
    assert rows[1]["predicted_bin"] == pytest.approx(0.9)
    assert rows[1]["observed_bin"] == pytest.approx(1.0)


def test_representative_selection_is_deterministic_and_tiered():
    metric = np.arange(30, dtype=float)
    first = select_representative_examples(metric, seed=9, per_tier=2)
    second = select_representative_examples(metric, seed=9, per_tier=2)
    assert first == second
    assert [row["tier"] for row in first] == [
        "good", "good", "intermediate", "intermediate", "poor", "poor"
    ]
    assert all(row["index"] < 10 for row in first[:2])
    assert all(10 <= row["index"] < 20 for row in first[2:4])
    assert all(row["index"] >= 20 for row in first[4:])


def test_checkpoint_split_resolution_defaults_to_test_and_never_falls_back():
    checkpoint = {"splits": {"train": [0, 1], "val": [2], "test": [3, 4]}}
    name, indices = resolve_rbpnet_checkpoint_indices(
        checkpoint, split=None, n_examples=5
    )
    assert name == "test"
    assert indices == [3, 4]
    assert resolve_rbpnet_checkpoint_indices(
        checkpoint, split="all", n_examples=5
    ) == ("all", [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="will not fall back"):
        resolve_rbpnet_checkpoint_indices({}, split="test", n_examples=5)
    with pytest.raises(ValueError, match="both train and test"):
        resolve_rbpnet_checkpoint_indices(
            {"splits": {"train": [0], "val": [1], "test": [0]}},
            split="test",
            n_examples=2,
        )
