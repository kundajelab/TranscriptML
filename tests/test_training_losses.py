import math

import numpy as np
import pytest
import torch

import transcriptml.training.trainer as trainer
from transcriptml.data.bundle import DatasetBundle
from transcriptml.training.losses import build_training_loss
from transcriptml.training.trainer import train_model


def test_weighted_mse_from_se_uses_clipped_inverse_variance():
    metadata = [{"se": "0.01"}, {"se": "2.0"}, {"se": "100.0"}]
    loss_fn, aux, normalized = build_training_loss(
        {
            "name": "weighted_mse",
            "se_col": "se",
            "eps": 1e-12,
            "min_weight": 0.1,
            "max_weight": 10.0,
        },
        metadata=metadata,
        n_examples=3,
    )

    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([0.0, 1.0, 1.0])
    out = loss_fn(y_pred, y_true, {"weight": torch.as_tensor(aux["weight"])})

    weights = np.array([10.0, 0.25, 0.1], dtype=np.float32)
    expected = float(np.sum(weights * np.array([1.0, 1.0, 4.0])) / np.sum(weights))
    assert normalized["name"] == "weighted_mse"
    assert np.allclose(aux["weight"], weights)
    assert torch.isclose(out.loss, torch.tensor(expected))


def test_weighted_mse_requires_one_weight_source():
    with pytest.raises(ValueError, match="exactly one"):
        build_training_loss(
            {"name": "weighted_mse", "weight_col": "w", "se_col": "se"},
            metadata=[{"w": 1.0, "se": 0.2}],
            n_examples=1,
        )


def test_binomial_nll_matches_expected_per_read_cross_entropy():
    metadata = [{"total_reads": "10", "new_reads": "3", "pulse_hours": "2.0"}]
    loss_fn, aux, normalized = build_training_loss(
        {"name": "binomial_nll"},
        metadata=metadata,
        n_examples=1,
    )

    y_pred = torch.tensor([math.log(0.2)])
    out = loss_fn(
        y_pred,
        torch.tensor([0.0]),
        {name: torch.as_tensor(values) for name, values in aux.items()},
    )

    rate_time = 0.2 * 2.0
    p_new = 1.0 - math.exp(-rate_time)
    expected = -(3.0 * math.log(p_new) + 7.0 * (-rate_time)) / 10.0
    assert normalized["name"] == "binomial_nll"
    assert torch.isclose(out.loss, torch.tensor(expected, dtype=out.loss.dtype))


def test_binomial_nll_validates_count_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        build_training_loss(
            {"name": "binomial_nll"},
            metadata=[{"total_reads": 5, "new_reads": 6, "pulse_hours": 1.0}],
            n_examples=1,
        )


def _tiny_x(n: int) -> np.ndarray:
    x = np.zeros((n, 4, 12), dtype=np.uint8)
    for i in range(n):
        x[i, i % 4, :] = 1
    return x


def _tiny_model_config() -> dict[str, object]:
    return {
        "name": "small_cnn",
        "params": {
            "in_ch": 4,
            "n_filters": 4,
            "kernel_size": 3,
            "n_layers": 1,
            "dropout": 0.0,
            "head_hidden": 4,
        },
    }


class _BatchNormRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(4)
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float().mean(dim=-1)
        return self.fc(self.bn(z)).squeeze(-1)


def test_train_model_default_mse_remains_compatible(tmp_path):
    bundle = DatasetBundle(
        X=_tiny_x(6),
        y=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        schema="rna4",
        splits={"train": [0, 1, 2, 3], "val": [4], "test": [5]},
    )

    result = train_model(
        bundle,
        {
            "dataset": "unused",
            "output_dir": str(tmp_path),
            "model": _tiny_model_config(),
            "batch_size": 2,
            "epochs": 1,
            "patience": 0,
            "progress": False,
        },
    )

    assert result["summary"]["loss"] == {"name": "mse"}
    assert result["summary"]["test_loss"] == pytest.approx(result["summary"]["test_mse"])
    assert (tmp_path / "best.pt").exists()


def test_train_model_drops_singleton_training_batch_for_batchnorm(tmp_path, monkeypatch):
    bundle = DatasetBundle(
        X=_tiny_x(8),
        y=np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        schema="rna4",
        splits={"train": [0, 1, 2, 3, 4], "val": [5, 6], "test": [7]},
    )
    monkeypatch.setattr(trainer, "build_model", lambda _config: _BatchNormRegressionModel())

    result = train_model(
        bundle,
        {
            "dataset": "unused",
            "output_dir": str(tmp_path),
            "model": _tiny_model_config(),
            "batch_size": 2,
            "epochs": 1,
            "patience": 0,
            "progress": False,
        },
    )

    assert np.isfinite(result["history"][0]["train_loss"])
    assert (tmp_path / "best.pt").exists()


def test_train_model_binomial_nll_can_train_without_targets(tmp_path):
    metadata = [
        {"total_reads": 10 + i, "new_reads": 2 + i % 3, "pulse_hours": 2.0}
        for i in range(6)
    ]
    bundle = DatasetBundle(
        X=_tiny_x(6),
        y=None,
        schema="rna4",
        metadata=metadata,
        splits={"train": [0, 1, 2, 3], "val": [4], "test": [5]},
    )

    result = train_model(
        bundle,
        {
            "dataset": "unused",
            "output_dir": str(tmp_path),
            "model": _tiny_model_config(),
            "batch_size": 2,
            "epochs": 1,
            "patience": 0,
            "progress": False,
            "loss": {"name": "binomial_nll"},
        },
    )

    assert result["summary"]["loss"]["name"] == "binomial_nll"
    assert result["summary"]["test_mse"] is None
    assert math.isnan(result["history"][0]["train_pearson"])
