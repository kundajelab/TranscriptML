from __future__ import annotations

import json

import pytest
import torch

from transcriptml.cli.main import main
from transcriptml.models.cnn import SmallCNN
from transcriptml.models.registry import build_model, load_checkpoint, model_default_params, save_checkpoint
from transcriptml.models.reproduce import SalukiExact
from transcriptml.models.saluki_cnn import SalukiCNNHybrid


def _small_sections(**fusion_overrides):
    fusion = {
        "head_hidden": 8,
        "head_norm": "layer",
        "head_dropout": 0.0,
        "projection_dropout": 0.0,
    }
    fusion.update(fusion_overrides)
    return {
        "saluki": {
            "filters": 8,
            "kernel_size": 3,
            "num_layers": 1,
            "dropout": 0.0,
            "augment_shift": 0,
        },
        "cnn": {
            "n_filters": 8,
            "kernel_size": 3,
            "n_layers": 1,
            "dropout": 0.0,
        },
        "fusion": fusion,
    }


def test_registry_defaults_and_cli_are_nested_json(capsys):
    defaults = model_default_params("saluki_cnn_hybrid")
    assert set(defaults) == {"saluki", "cnn", "fusion"}
    assert defaults["saluki"]["seq_depth"] == 6
    assert defaults["cnn"]["in_ch"] == 6
    assert defaults["fusion"]["mode"] == "concat"
    json.dumps(defaults)

    main(["models", "show", "saluki_cnn_hybrid", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"name": "saluki_cnn_hybrid", "params": defaults}


def test_partial_nested_sections_merge_with_defaults():
    model = build_model(
        {
            "name": "saluki_cnn_hybrid",
            "params": {
                "saluki": {"filters": 16},
                "cnn": {"n_filters": 12},
                "fusion": {"head_hidden": 10},
            },
        }
    )
    assert model.saluki_encoder.output_dim == 16
    assert len(model.saluki_encoder.blocks) == 6
    assert model.cnn_encoder.output_dim == 24
    assert len(model.cnn_encoder.encoder) == 8
    assert model.head.linear1.out_features == 10


def test_default_model_has_expected_branch_and_fusion_widths():
    model = SalukiCNNHybrid().eval()
    x = torch.randn(2, 6, 1024)
    saluki_features, cnn_features = model.forward_branch_features(x)
    assert saluki_features.shape == (2, 64)
    assert cnn_features.shape == (2, 128)
    assert model.forward_features(x).shape == (2, 192)
    assert model(x).shape == (2,)


def test_small_cnn_forward_features_preserves_forward_computation():
    model = SmallCNN(in_ch=6, n_filters=8, kernel_size=3, n_layers=1, dropout=0.0).eval()
    x = torch.randn(3, 6, 32)
    features = model.forward_features(x)
    assert features.shape == (3, 16)
    assert torch.allclose(model(x), model.head(features).squeeze(-1))


def test_raw_branch_embeddings_match_standalone_models():
    sections = _small_sections()
    torch.manual_seed(31)
    standalone_saluki = SalukiExact(
        seq_depth=6,
        filters=8,
        kernel_size=3,
        num_layers=1,
        dropout=0.0,
        augment_shift=0,
    ).eval()
    torch.manual_seed(31)
    hybrid = SalukiCNNHybrid(**sections).eval()

    standalone_cnn = SmallCNN(
        in_ch=6,
        n_filters=8,
        kernel_size=3,
        n_layers=1,
        dropout=0.0,
    ).eval()
    standalone_cnn.encoder.load_state_dict(hybrid.cnn_encoder.encoder.state_dict())

    x = torch.randn(2, 6, 64)
    saluki_features, cnn_features = hybrid.forward_branch_features(x)
    assert torch.allclose(saluki_features, standalone_saluki.forward_features(x), atol=1e-6)
    assert torch.allclose(cnn_features, standalone_cnn.forward_features(x), atol=1e-6)
    assert hybrid.forward_features(x).shape == (2, 24)
    assert torch.allclose(
        hybrid.forward_features(x),
        torch.cat([saluki_features, cnn_features], dim=1),
        atol=1e-6,
    )


def test_shared_shift_feeds_identical_inputs_to_both_branches():
    sections = _small_sections()
    sections["saluki"]["augment_shift"] = 3
    model = SalukiCNNHybrid(**sections)
    captured: dict[str, torch.Tensor] = {}

    def capture(name):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()

        return hook

    saluki_hook = model.saluki_encoder.conv0.register_forward_pre_hook(capture("saluki"))
    cnn_hook = model.cnn_encoder.encoder[0].register_forward_pre_hook(capture("cnn"))
    x = torch.randn(2, 6, 64)
    try:
        model.train()
        torch.manual_seed(101)
        model.forward_branch_features(x)
        assert torch.equal(captured["saluki"], captured["cnn"])

        model.eval()
        model.forward_branch_features(x)
        assert torch.equal(captured["saluki"], x)
        assert torch.equal(captured["cnn"], x)
    finally:
        saluki_hook.remove()
        cnn_hook.remove()


def test_projected_concat_supports_unequal_widths_and_gradients():
    sections = _small_sections(
        mode="project_concat",
        saluki_projection_dim=5,
        cnn_projection_dim=7,
        projection_norm="layer",
        projection_activation="silu",
        output_dim=3,
    )
    model = SalukiCNNHybrid(**sections)
    x = torch.randn(2, 6, 64, requires_grad=True)
    assert model.fused_dim == 12
    assert model.forward_features(x).shape == (2, 12)
    prediction = model(x)
    assert prediction.shape == (2, 3)
    prediction.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.saluki_projection.linear.weight.grad is not None
    assert model.cnn_projection.linear.weight.grad is not None


def test_input_layouts_and_checkpoint_roundtrip(tmp_path):
    config = {"name": "saluki_cnn_hybrid", "params": _small_sections()}
    model = build_model(config).eval()
    x = torch.randn(2, 6, 64)
    expected = model(x)
    assert expected.shape == (2,)
    assert torch.allclose(expected, model(x.transpose(1, 2)), atol=1e-6)

    checkpoint_path = tmp_path / "hybrid.pt"
    save_checkpoint(checkpoint_path, model, config)
    loaded, checkpoint = load_checkpoint(checkpoint_path)
    assert isinstance(loaded, SalukiCNNHybrid)
    assert checkpoint["model_config"] == config
    assert torch.allclose(expected, loaded(x), atol=1e-6)


def test_head_defaults_and_layernorm_singleton_training():
    default_model = SalukiCNNHybrid(**_small_sections(head_norm="batch"))
    assert isinstance(default_model.head.norm1, torch.nn.BatchNorm1d)
    assert isinstance(default_model.head.norm2, torch.nn.BatchNorm1d)
    assert default_model.head.norm1.eps == pytest.approx(0.001)
    assert default_model.head.norm1.momentum == pytest.approx(0.1)

    layer_model = SalukiCNNHybrid(**_small_sections(head_norm="layer")).train()
    assert isinstance(layer_model.head.norm1, torch.nn.LayerNorm)
    assert layer_model(torch.randn(1, 6, 64)).shape == (1,)


def test_saluki_initialization_applies_to_fusion_but_not_discarded_heads():
    model = SalukiCNNHybrid(
        **_small_sections(
            mode="project_concat",
            weight_init="saluki",
            saluki_projection_dim=5,
            cnn_projection_dim=7,
        )
    )
    assert torch.count_nonzero(model.saluki_projection.linear.bias) == 0
    assert torch.count_nonzero(model.cnn_projection.linear.bias) == 0
    assert torch.count_nonzero(model.head.linear1.bias) == 0
    assert torch.count_nonzero(model.head.linear2.bias) == 0

    state_keys = set(model.state_dict())
    assert not any(key.startswith("saluki_encoder.bn") for key in state_keys)
    assert not any(key.startswith("saluki_encoder.fc") for key in state_keys)
    assert not any(key.startswith("cnn_encoder.head") for key in state_keys)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("saluki", {"unknown": 1}, "Unknown saluki parameter"),
        ("cnn", {"unknown": 1}, "Unknown cnn parameter"),
        ("fusion", {"unknown": 1}, "Unknown fusion parameter"),
        ("cnn", {"in_ch": 4}, "seq_depth and cnn.in_ch must match"),
        ("fusion", {"mode": "gated"}, "fusion.mode must be one of"),
        ("fusion", {"saluki_projection_dim": 0}, "must be positive"),
        ("fusion", {"projection_dropout": 1.0}, "must be in"),
        ("fusion", {"head_norm": "group"}, "head_norm must be one of"),
    ],
)
def test_invalid_nested_configs_raise_clear_errors(section, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        SalukiCNNHybrid(**{section: value})
