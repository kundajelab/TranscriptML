import numpy as np
import pytest
import torch

from transcriptml.data.bundle import DatasetBundle
from transcriptml.models.registry import build_model
from transcriptml.training.trainer import TrainConfig, _monitor_improved, _monitor_names, _select_splits
from transcriptml.training.splits import predefined_split_indices, random_split_indices


def test_random_split_reproducible_no_overlap():
    a = random_split_indices(20, val_frac=0.2, test_frac=0.2, seed=7)
    b = random_split_indices(20, val_frac=0.2, test_frac=0.2, seed=7)
    assert a == b
    sets = [set(a[k]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 20


def test_predefined_split_from_metadata():
    splits = predefined_split_indices(
        [{"split": "train"}, {"split": "val"}, {"split": "test"}, {"split": "validation"}]
    )
    assert splits["train"] == [0]
    assert splits["val"] == [1, 3]
    assert splits["test"] == [2]


def test_training_split_source_auto_bundle_and_config(tmp_path):
    bundle = DatasetBundle(
        X=np.zeros((4, 4, 3), dtype=np.float32),
        y=np.arange(4, dtype=np.float32),
        splits={"train": [0, 1], "val": [2], "test": [3]},
    )
    cfg = TrainConfig(
        dataset="unused",
        output_dir=str(tmp_path),
        split_source="auto",
        split={"method": "predefined", "splits": {"train": [3], "val": [2], "test": [1]}},
    )

    splits, source = _select_splits(bundle, cfg)
    assert source == "bundle"
    assert splits["train"] == [0, 1]

    cfg.split_source = "config"
    splits, source = _select_splits(bundle, cfg)
    assert source == "config"
    assert splits["train"] == [3]


def test_training_split_source_bundle_requires_bundle(tmp_path):
    bundle = DatasetBundle(X=np.zeros((4, 4, 3), dtype=np.float32), y=np.arange(4, dtype=np.float32))
    cfg = TrainConfig(dataset="unused", output_dir=str(tmp_path), split_source="bundle")
    with pytest.raises(ValueError, match="has no splits"):
        _select_splits(bundle, cfg)


def test_multiple_monitor_metrics_use_or_improvement():
    monitors = _monitor_names(["val_loss", "val_pearson"])
    best = {"val_loss": 1.0, "val_pearson": 0.5}

    improved, values = _monitor_improved({"val_loss": 1.1, "val_pearson": 0.6}, monitors, best)
    assert improved
    assert values == {"val_loss": 1.1, "val_pearson": 0.6}

    improved, _ = _monitor_improved({"val_loss": 0.9, "val_pearson": 0.4}, monitors, best)
    assert improved

    improved, _ = _monitor_improved({"val_loss": 1.1, "val_pearson": 0.4}, monitors, best)
    assert not improved


def test_model_registry_dummy_forward():
    x4 = torch.randn(2, 4, 32)
    small = build_model({"name": "small_cnn", "params": {"in_ch": 4, "n_filters": 8, "head_hidden": 8}})
    assert small(x4).shape == (2,)

    x6 = torch.randn(2, 6, 32)
    saluki = build_model(
        {
            "name": "saluki_gru",
            "params": {"in_ch": 6, "base_ch": 8, "n_convs": 2, "gru_hidden": 8, "head_hidden": 8},
        }
    )
    assert saluki(x6).shape == (2,)

    legnet = build_model(
        {
            "name": "legnet",
            "params": {
                "in_ch": 4,
                "stem_ch": 8,
                "ef_block_sizes": [8],
                "pool_sizes": [2],
                "resize_factor": 2,
            },
        }
    )
    legnet.eval()
    assert legnet(x4).shape == (2,)
