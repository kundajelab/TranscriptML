import numpy as np
import torch

from transcriptml.models.registry import build_model
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
