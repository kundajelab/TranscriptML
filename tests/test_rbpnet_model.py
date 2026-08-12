import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from transcriptml.data.bundle import DatasetBundle, save_bundle
from transcriptml.models.rbpnet import (
    RBPNet,
    SameLengthConvTranspose1d,
    SamePadConv1d,
)
from transcriptml.models.registry import build_model, load_checkpoint
from transcriptml.rbpnet.dataset import RBPNetDataset, collate_rbpnet
from transcriptml.rbpnet.evaluation import evaluate_rbpnet_report
from transcriptml.rbpnet.losses import (
    RBPNetObjective,
    multinomial_nll,
    replicate_binomial_nll,
)
from transcriptml.training.splits import group_split_indices, validate_group_disjoint
from transcriptml.training.evaluation import evaluate_checkpoint
from transcriptml.training.trainer import train_model
from transcriptml.workflows.chromosome_cv import (
    create_chromosome_cv_plan,
    load_chromosome_cv_plan,
    save_chromosome_cv_plan,
)


def _synthetic_bundle(n=9, length=16, jitter=2):
    width = length + 2 * jitter
    rng = np.random.default_rng(8)
    bases = rng.integers(0, 4, size=(n, width))
    X = np.zeros((n, 4, width), dtype=np.uint8)
    for index in range(n):
        X[index, bases[index], np.arange(width)] = 1
    ip = rng.poisson(0.5, size=(n, 2, width)).astype(np.uint32)
    sm = rng.poisson(0.7, size=(n, width)).astype(np.uint32)
    metadata = []
    selection_ip = np.zeros((n, 2), dtype=np.uint64)
    selection_sm = np.zeros(n, dtype=np.uint64)
    for index in range(n):
        if index == 0:
            anchor, materialized_start, selection = 1, 0, (0, 4)
        elif index == 1:
            anchor, materialized_start, selection = 98, 80, (96, 100)
        else:
            anchor = 30 + index * 3
            materialized_start = anchor - width // 2
            selection = (anchor - 2, anchor + 3 + index % 2)
        local_start = selection[0] - materialized_start
        local_end = selection[1] - materialized_start
        selection_ip[index] = ip[index, :, local_start:local_end].sum(axis=1)
        selection_sm[index] = sm[index, local_start:local_end].sum()
        metadata.append(
            {
                "example_id": f"ex{index}",
                "gene_id": f"g{index // 3}",
                "transcript_id": f"tx{index // 3}",
                "chromosome": "chr1",
                "strand": "-" if index % 2 else "+",
                "coordinate_space": "mature_transcript",
                "locus_length": 100,
                "transcript_anchor": anchor,
                "selection_start": selection[0],
                "selection_end": selection[1],
                "selection_state": ("peak", "gray", "negative")[index % 3],
                "region_type": ("cds", "3putr", "mixed")[index % 3],
                "sequence_materialized_start": materialized_start,
                "sequence_materialized_end": materialized_start + width,
                "profile_materialized_start": materialized_start,
                "profile_materialized_end": materialized_start + width,
                "group_gene_id": f"g{index // 3}",
                "group_transcript_id": f"tx{index // 3}",
                "group_chromosome": f"chr{index // 3 + 1}",
            }
        )
    arrays = {
        "sminput_profiles": sm,
        "ip_profiles": ip,
        "sequence_valid_mask": np.ones((n, width), dtype=np.uint8),
        "profile_valid_mask": np.ones((n, width), dtype=np.uint8),
        "profile_sminput_totals": sm.sum(axis=1, dtype=np.uint64),
        "profile_ip_totals": ip.sum(axis=2, dtype=np.uint64),
        "selection_sminput_counts": selection_sm,
        "selection_ip_counts": selection_ip,
    }
    return DatasetBundle(
        X=X,
        ids=[f"ex{index}" for index in range(n)],
        metadata=metadata,
        arrays=arrays,
        config={
            "bundle_format": "transcriptml-rbpnet-bundle",
            "bundle_format_version": "1",
            "coordinate_space": "mature_transcript",
            "input_length": length,
            "profile_length": length,
            "max_jitter": jitter,
            "materialized_sequence_length": width,
            "materialized_profile_length": width,
            "transcript_end_policy": "shift_to_fit",
            "sample_metadata": {
                "sminput": {"name": "sminput", "effective_library_size": 100},
                "ip": [
                    {"name": "ip1", "effective_library_size": 50},
                    {"name": "ip2", "effective_library_size": 200},
                ],
                "ip_axis_order": ["ip1", "ip2"],
            },
        },
    )


def _small_model(*, enrichment="linear"):
    return RBPNet(
        n_filters=8,
        initial_kernel_size=3,
        n_residual_blocks=1,
        residual_kernel_size=3,
        dilations=[1],
        normalization="none",
        dropout=0.0,
        profile_head_kernel_size=3,
        enrichment_head_type=enrichment,
        profile_length=16,
    )


def test_default_rbpnet_architecture_probabilities_and_receptive_field():
    model = RBPNet()
    assert len(model.residual_blocks) == 5
    assert model.dilations == (2, 4, 8, 16, 32)
    assert model.receptive_field == 322
    assert model.receptive_field_extents == (160, 161)
    assert model.target_profile_head.head_type == "transpose_conv"
    output = model(torch.randn(2, 4, 300))
    assert output.target_logits.shape == output.control_logits.shape == (2, 300)
    assert output.mixing_logit.shape == output.pi.shape == (2,)
    assert output.enrichment_logit is None
    torch.testing.assert_close(output.target_probs.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(output.control_probs.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(output.ip_probs.sum(dim=-1), torch.ones(2))
    assert torch.all((output.pi >= 0) & (output.pi <= 1))
    expected = (
        output.pi[:, None] * output.target_probs
        + (1 - output.pi[:, None]) * output.control_probs
    )
    torch.testing.assert_close(output.ip_probs, expected)


def test_architecture_override_same_length_and_indexed_alignment():
    model = build_model(
        {
            "name": "rbpnet",
            "params": {
                "n_filters": 7,
                "initial_kernel_size": 4,
                "n_residual_blocks": 2,
                "residual_kernel_size": 4,
                "dilations": [1, 3],
                "normalization": "layer",
                "dropout": 0.0,
                "profile_head_type": "conv",
                "profile_head_kernel_size": 6,
                "profile_length": 31,
                "enrichment_head_type": "mlp",
                "enrichment_hidden": 5,
            },
        }
    )
    output = model(torch.randn(3, 4, 31), measurement_mask=torch.ones(3, 31))
    assert model.receptive_field == 16
    assert output.ip_probs.shape == (3, 31)
    assert output.enrichment_logit.shape == (3,)

    conv = SamePadConv1d(1, 1, 12, bias=False)
    conv.conv.weight.data.zero_()
    conv.conv.weight.data[0, 0, conv.padding[0]] = 1
    impulse = torch.zeros(1, 1, 25)
    impulse[0, 0, 11] = 1
    torch.testing.assert_close(conv(impulse), impulse)

    transpose = SameLengthConvTranspose1d(1, 1, 6, bias=False)
    transpose.conv.weight.data.zero_()
    transpose.conv.weight.data[0, 0, transpose.crop[0]] = 1
    torch.testing.assert_close(transpose(impulse), impulse)

    residual_model = _small_model(enrichment="none")
    block = residual_model.residual_blocks[0]
    block.conv.conv.weight.data.zero_()
    if block.conv.conv.bias is not None:
        block.conv.conv.bias.data.zero_()
    hidden = torch.randn(2, 8, 16)
    torch.testing.assert_close(block(hidden), hidden)


def test_multinomial_and_binomial_likelihoods_zero_edges_and_offsets():
    probabilities = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    counts = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    result = multinomial_nll(probabilities.log(), counts)
    expected = -math.log(3 * 0.25 * 0.75**2)
    assert result.denominator.item() == 1
    assert result.loss.item() == pytest.approx(expected)
    empty = multinomial_nll(torch.log(torch.tensor([[0.5, 0.5]])), torch.zeros(1, 2))
    assert empty.loss.item() == 0
    assert empty.denominator.item() == 0

    eta = torch.tensor([0.0, math.log(2.0)])
    ip = torch.tensor([[0.0, 3.0], [4.0, 0.0]])
    sm = torch.tensor([3.0, 0.0])
    offsets = torch.tensor([0.0, math.log(2.0)])
    observed = replicate_binomial_nll(eta, ip, sm, offsets)
    logits = eta[:, None] + offsets[None, :]
    total = ip + sm[:, None]
    expected_values = -torch.distributions.Binomial(
        total_count=total, logits=logits
    ).log_prob(ip)
    valid = total > 0
    assert observed.loss.item() == pytest.approx(expected_values[valid].mean().item())
    assert logits[0, 0].item() == 0.0
    assert logits[0, 1].item() == pytest.approx(math.log(2.0))
    assert torch.isfinite(observed.per_observation).all()


def test_dataset_measurement_masks_jitter_boundaries_and_deterministic_eval():
    bundle = _synthetic_bundle()
    training = RBPNetDataset(bundle, max_train_jitter=2, training=True, seed=11)
    left_minus = training.item_for_shift(0, -2)
    left_plus = training.item_for_shift(0, 2)
    assert left_minus["crop_start"] == left_plus["crop_start"] == 0
    assert left_minus["measurement_mask"].sum() == 4
    right = training.item_for_shift(1, 2)
    assert right["crop_start"] == 84
    assert right["measurement_mask"].sum() == 4
    variable = training.item_for_shift(3, 0)
    assert variable["measurement_mask"].sum() == 6
    np.testing.assert_array_equal(
        left_minus["pooled_ip_profile"],
        left_minus["individual_ip_profiles"].sum(axis=0),
    )
    assert training.depth_offsets.tolist() == pytest.approx(
        [math.log(0.5), math.log(2.0)]
    )

    evaluation = RBPNetDataset(bundle, max_train_jitter=0, training=False)
    first = evaluation[4]
    second = evaluation[4]
    assert first["jitter_shift"] == second["jitter_shift"] == 0
    np.testing.assert_array_equal(first["sequence"], second["sequence"])
    training.set_epoch(7)
    a = training[4]
    training.set_epoch(7)
    b = training[4]
    assert a["jitter_shift"] == b["jitter_shift"]
    np.testing.assert_array_equal(a["sequence"], b["sequence"])

    interior_left = training.item_for_shift(4, -2)
    interior_right = training.item_for_shift(4, 2)
    materialized_start = bundle.metadata[4]["sequence_materialized_start"]
    left_offset = int(interior_left["crop_start"]) - materialized_start
    right_offset = int(interior_right["crop_start"]) - materialized_start
    assert (left_offset, right_offset) == (0, 4)
    np.testing.assert_array_equal(
        interior_left["sequence"], bundle.X[4, :, left_offset : left_offset + 16]
    )
    np.testing.assert_array_equal(
        interior_right["individual_ip_profiles"],
        bundle.arrays["ip_profiles"][4, :, right_offset : right_offset + 16],
    )
    assert np.flatnonzero(interior_left["measurement_mask"])[0] == (
        np.flatnonzero(interior_right["measurement_mask"])[0] + 4
    )


def test_enrichment_pooling_uses_weighted_variable_measurement_mask():
    model = _small_model(enrichment="linear")
    hidden = torch.arange(2 * 8 * 16, dtype=torch.float32).reshape(2, 8, 16)
    masks = torch.zeros(2, 16)
    masks[0, 2:5] = 1
    masks[1, 4:10] = torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.float32)
    pooled = model._pool_measurement(hidden, masks)
    torch.testing.assert_close(pooled[0], hidden[0, :, 2:5].mean(dim=-1))
    expected_second = (hidden[1, :, 4:10] * masks[1, 4:10]).sum(dim=-1) / 8
    torch.testing.assert_close(pooled[1], expected_second)


def test_loss_pools_ip_replicates_and_gradients_reach_every_enabled_head():
    dataset = RBPNetDataset(_synthetic_bundle(), max_train_jitter=0, training=False)
    batch = collate_rbpnet([dataset[index] for index in range(3)])
    model = _small_model(enrichment="linear")
    output = model(
        batch.sequence,
        measurement_mask=batch.measurement_mask,
        profile_mask=batch.profile_valid_mask,
    )
    objective = RBPNetObjective(enrichment_enabled=True)
    loss = objective(output, batch)
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    for module in (
        model.initial_conv,
        model.target_profile_head,
        model.control_profile_head,
        model.mixing_head,
        model.enrichment_head,
    ):
        assert module is not None
        assert any(parameter.grad is not None for parameter in module.parameters())

    # Enrichment consumes eta and known offsets only; changing pi leaves its
    # likelihood unchanged, i.e. there is deliberately no pi-BNLL.
    enrichment_a = objective(output, batch).components["enrichment_loss"]
    changed_pi = replace(
        output,
        mixing_logit=torch.full_like(output.mixing_logit, 100.0),
        pi=torch.ones_like(output.pi),
    )
    enrichment_b = objective(changed_pi, batch).components["enrichment_loss"]
    torch.testing.assert_close(enrichment_a, enrichment_b)


def test_group_split_and_structured_training_profile_only_and_enrichment(tmp_path):
    bundle = _synthetic_bundle()
    splits = group_split_indices(
        bundle.metadata, group_col="group_gene_id", val_frac=0.2, test_frac=0.2, seed=3
    )
    validate_group_disjoint(splits, bundle.metadata, group_col="group_gene_id")
    owners = {}
    for split, indices in splits.items():
        for index in indices:
            group = bundle.metadata[index]["group_gene_id"]
            assert group not in owners or owners[group] == split
            owners[group] = split

    base_config = {
        "dataset": "unused",
        "batch_size": 3,
        "epochs": 1,
        "patience": 0,
        "progress": False,
        "learning_rate": 0.005,
        "max_train_jitter": 2,
        "model": {
            "name": "rbpnet",
            "params": {
                "n_filters": 8,
                "initial_kernel_size": 3,
                "n_residual_blocks": 1,
                "residual_kernel_size": 3,
                "dilations": [1],
                "normalization": "none",
                "dropout": 0.0,
                "profile_head_kernel_size": 3,
                "profile_length": 16,
            },
        },
        "loss": {"name": "rbpnet"},
        "split_source": "config",
        "split": {
            "method": "group",
            "group_col": "group_gene_id",
            "val_frac": 0.2,
            "test_frac": 0.2,
            "seed": 3,
        },
    }
    profile_config = dict(base_config)
    profile_config["output_dir"] = str(tmp_path / "profile")
    profile_config["model"] = {
        "name": "rbpnet",
        "params": {**base_config["model"]["params"], "enrichment_head_type": "none"},
    }
    profile = train_model(bundle, profile_config)
    assert profile["summary"]["trainer"] == "rbpnet"
    assert profile["summary"]["loss"]["effective_lambda_enrichment"] == 0
    assert np.isfinite(profile["history"][0]["train_loss"])

    enrichment_config = dict(base_config)
    enrichment_config["output_dir"] = str(tmp_path / "enrichment")
    enrichment_config["model"] = {
        "name": "rbpnet",
        "params": {**base_config["model"]["params"], "enrichment_head_type": "linear"},
    }
    enriched = train_model(bundle, enrichment_config)
    assert enriched["summary"]["loss"]["effective_lambda_enrichment"] == 1
    assert np.isfinite(enriched["history"][0]["train_enrichment_loss"])
    assert (tmp_path / "enrichment" / "best.pt").is_file()
    assert (tmp_path / "enrichment" / "test_predictions.csv").is_file()

    bundle_dir = tmp_path / "bundle"
    # Deliberately disagree with the checkpoint: RBPNet evaluation must use
    # the immutable training artifact and never silently use bundle.splits.
    bundle.splits = {
        "train": list(range(1, len(bundle.ids) - 1)),
        "val": [len(bundle.ids) - 1],
        "test": [0],
    }
    save_bundle(bundle, bundle_dir)
    predictions_path = tmp_path / "checkpoint_predictions.csv"
    evaluated = evaluate_checkpoint(
        tmp_path / "enrichment" / "best.pt",
        bundle_dir,
        predictions_path,
        batch_size=3,
        progress=False,
    )
    _, saved_checkpoint = load_checkpoint(
        tmp_path / "enrichment" / "best.pt", map_location="cpu"
    )
    checkpoint_test = saved_checkpoint["splits"]["test"]
    assert evaluated["indices"] == checkpoint_test
    assert evaluated["pi"].shape == (len(checkpoint_test),)
    assert evaluated["enrichment_logit"].shape == (len(checkpoint_test),)
    assert predictions_path.is_file()

    report_dir = tmp_path / "evaluation_report"
    report = evaluate_checkpoint(
        tmp_path / "enrichment" / "best.pt",
        bundle_dir,
        out_dir=report_dir,
        batch_size=3,
        save_profiles=True,
        representative_per_tier=1,
        progress=False,
    )
    assert report["indices"] == checkpoint_test
    assert report["indices"] != bundle.splits["test"]
    for name in (
        "summary.json",
        "examples.parquet",
        "stratified_metrics.parquet",
        "calibration.parquet",
    ):
        assert (report_dir / name).is_file()
    assert (report_dir / "plots" / "profile_performance_vs_read_depth.png").is_file()
    assert (report_dir / "plots" / "enrichment_calibration.png").is_file()
    assert (
        report_dir / "plots" / "stratified_performance_summaries.png"
    ).is_file()
    predicted_ip = np.load(
        report_dir / "predicted_ip_profiles.npy", mmap_mode="r"
    )
    assert isinstance(predicted_ip, np.memmap)
    assert predicted_ip.shape == (len(checkpoint_test), 16)

    profile_report_dir = tmp_path / "profile_evaluation_report"
    evaluate_checkpoint(
        tmp_path / "profile" / "best.pt",
        bundle_dir,
        out_dir=profile_report_dir,
        representative_per_tier=0,
        progress=False,
    )
    import pyarrow.parquet as pq

    assert pq.read_table(profile_report_dir / "calibration.parquet").num_rows == 0
    assert not (profile_report_dir / "plots" / "enrichment_calibration.png").exists()

    unsafe = dict(base_config)
    unsafe["output_dir"] = str(tmp_path / "unsafe")
    unsafe["split"] = {"method": "random", "val_frac": 0.2, "test_frac": 0.2}
    with pytest.raises(ValueError, match="row-level random splitting is unsafe"):
        train_model(bundle, unsafe)


def test_tiny_batch_can_overfit():
    torch.manual_seed(4)
    dataset = RBPNetDataset(_synthetic_bundle(n=3), max_train_jitter=0, training=False)
    batch = collate_rbpnet([dataset[index] for index in range(3)])
    model = _small_model(enrichment="linear")
    objective = RBPNetObjective(enrichment_enabled=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    def step(update):
        output = model(
            batch.sequence,
            measurement_mask=batch.measurement_mask,
            profile_mask=batch.profile_valid_mask,
        )
        value = objective(output, batch).loss
        if update:
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
        return float(value.detach())

    initial = step(False)
    for _ in range(25):
        step(True)
    final = step(False)
    assert final < initial


def test_rbpnet_report_gracefully_handles_one_replicate_and_optional_metadata(
    tmp_path,
):
    original = _synthetic_bundle(n=4)
    arrays = dict(original.arrays)
    arrays["ip_profiles"] = arrays["ip_profiles"][:, :1, :]
    arrays["profile_ip_totals"] = arrays["profile_ip_totals"][:, :1]
    arrays["selection_ip_counts"] = arrays["selection_ip_counts"][:, :1]
    config = dict(original.config)
    config["sample_metadata"] = {
        "sminput": {"name": "sminput", "effective_library_size": 100},
        "ip": [{"name": "ip1", "effective_library_size": 50}],
        "ip_axis_order": ["ip1"],
    }
    metadata = [
        {
            key: value
            for key, value in row.items()
            if key not in {"selection_state", "region_type"}
        }
        for row in original.metadata
    ]
    bundle = DatasetBundle(
        X=original.X,
        ids=original.ids,
        metadata=metadata,
        arrays=arrays,
        config=config,
    )
    result = evaluate_rbpnet_report(
        _small_model(enrichment="none"),
        {"splits": {"train": [0, 1], "val": [2], "test": [3]}},
        bundle,
        tmp_path / "report",
        representative_per_tier=0,
        progress=False,
    )
    assert result["indices"] == [3]
    summary = result["summary"]
    assert summary["enrichment_head_enabled"] is False
    assert "eta_vs_empirical_enrichment.png" in summary["plots"]["skipped"]
    assert "stratified_performance_summaries.png" in summary["plots"]["skipped"]
    assert not any(
        row["track"] == "replicate_ceiling"
        for row in summary["overall_metrics"]
    )


def test_rbpnet_training_consumes_saved_chromosome_cv_plan(tmp_path):
    bundle = _synthetic_bundle()
    plan_path = save_chromosome_cv_plan(
        create_chromosome_cv_plan(
            bundle.metadata, n_folds=3, group_col="group_chromosome"
        ),
        tmp_path / "cv3.json",
    )
    result = train_model(
        bundle,
        {
            "dataset": "unused",
            "output_dir": str(tmp_path / "fold1"),
            "batch_size": 3,
            "epochs": 1,
            "patience": 0,
            "progress": False,
            "learning_rate": 0.005,
            "model": {
                "name": "rbpnet",
                "params": {
                    "n_filters": 8,
                    "initial_kernel_size": 3,
                    "n_residual_blocks": 1,
                    "residual_kernel_size": 3,
                    "dilations": [1],
                    "normalization": "none",
                    "dropout": 0.0,
                    "profile_head_kernel_size": 3,
                    "profile_length": 16,
                    "enrichment_head_type": "none",
                },
            },
            "loss": {"name": "rbpnet"},
            "cv_plan": str(plan_path),
            "fold": 1,
        },
    )
    assert result["summary"]["split_source_used"] == "cv_plan"
    assert result["summary"]["cv_plan_id"] == load_chromosome_cv_plan(
        plan_path
    ).plan_id
    assert result["summary"]["fold"] == 1
    assert result["summary"]["split_group_col"] == "group_chromosome"
    assert result["summary"]["split_counts"] == {"train": 3, "val": 3, "test": 3}
