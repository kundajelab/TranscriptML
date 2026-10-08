from __future__ import annotations

import json

import pytest
import torch

from transcriptml.cli.main import main
from transcriptml.models.registry import build_model, load_checkpoint, model_default_params, save_checkpoint
from transcriptml.models.reproduce import SalukiExact, SalukiExactConfig
from transcriptml.models.universe import (
    ConvNeXtBlock1D,
    ConvolutionUniverse,
    ConvolutionUniverseConfig,
    StandardConvBlock1D,
)


def _small_params(**overrides):
    params = {
        "filters": 8,
        "num_layers": 2,
        "stem_kernel_size": 3,
        "kernel_size": 3,
        "dilations": [1, 2],
        "dropout": 0.0,
        "head_hidden": 8,
        "head_dropout": 0.0,
    }
    params.update(overrides)
    return params


def test_registry_defaults_and_cli_are_json_serializable(capsys):
    defaults = model_default_params("convolution_universe")
    assert defaults["in_ch"] == 6
    assert defaults["rnn_type"] == "none"
    assert defaults["dilations"] == [1, 2, 4, 8, 16, 32]
    json.dumps(defaults)

    main(["models", "show", "convolution_universe", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"name": "convolution_universe", "params": defaults}


def test_default_architecture_is_dilated_residual_cnn():
    model = ConvolutionUniverse()
    assert model.rnn is None
    assert model.receptive_field == 389
    assert model.downsample_factor == 1
    assert model.output_length(128) == 128
    assert all(
        isinstance(stage["block"], StandardConvBlock1D) for stage in model.tower.stages
    )


def test_input_layouts_output_dimensions_and_checkpoint_roundtrip(tmp_path):
    config = {
        "name": "convolution_universe",
        "params": _small_params(output_dim=3),
    }
    model = build_model(config).eval()
    x = torch.randn(2, 6, 32)
    expected = model(x)
    assert expected.shape == (2, 3)
    assert torch.allclose(expected, model(x.transpose(1, 2)))

    checkpoint_path = tmp_path / "universe.pt"
    save_checkpoint(checkpoint_path, model, config)
    loaded, checkpoint = load_checkpoint(checkpoint_path)
    assert isinstance(loaded, ConvolutionUniverse)
    assert checkpoint["model_config"] == config
    assert torch.allclose(expected, loaded(x))


def test_per_layer_channels_pooling_and_receptive_field():
    model = ConvolutionUniverse(
        in_ch=4,
        filters=[4, 8, 12],
        num_layers=3,
        stem_kernel_size=3,
        kernel_size=[3, 3, 3],
        dilations=[1, 1, 1],
        pooling=["none", "max", "average"],
        pool_size=[2, 2, 2],
        dropout=0.0,
        head_type="linear",
    ).eval()
    assert model.output_length(64) == 16
    assert model.downsample_factor == 4
    assert model.receptive_field == 14
    assert model(torch.randn(2, 4, 64)).shape == (2,)
    assert isinstance(model.tower.stages[1]["block"].shortcut, torch.nn.Conv1d)


def test_pooling_only_cnn_and_global_pooling_modes():
    x = torch.randn(2, 6, 64)
    for global_pool in ("avg", "max", "avgmax"):
        model = ConvolutionUniverse(
            **_small_params(
                dilations=1,
                pooling="max",
                pool_size=2,
                global_pool=global_pool,
                global_pool_size=2,
                head_type="linear",
            )
        ).eval()
        assert model.output_length(64) == 16
        assert model.downsample_factor == 4
        assert model(x).shape == (2,)


def test_convnext_tower_supports_width_changes_and_backward():
    model = ConvolutionUniverse(
        **_small_params(
            filters=[8, 12],
            block_type="convnext",
            drop_path_rate=0.1,
            output_dim=2,
        )
    )
    assert all(isinstance(stage["block"], ConvNeXtBlock1D) for stage in model.tower.stages)
    x = torch.randn(2, 6, 32, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 2)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
@pytest.mark.parametrize("summary", ["last", "mean", "max"])
def test_recurrent_variants(rnn_type, summary):
    model = ConvolutionUniverse(
        **_small_params(
            rnn_type=rnn_type,
            rnn_hidden=5,
            rnn_layers=2,
            rnn_bidirectional=True,
            rnn_reverse=True,
            rnn_summary=summary,
            rnn_dropout=0.1,
            pre_rnn_norm="layer",
            pre_rnn_activation="relu",
            head_type="linear",
        )
    ).eval()
    features = model.forward_features(torch.randn(2, 6, 32))
    assert features.shape == (2, 10)
    assert model(torch.randn(2, 6, 32)).shape == (2,)


def _assert_saluki_parameters_match(exact, universe):
    assert torch.equal(exact.conv0.weight, universe.tower.stem.weight)
    for exact_block, stage in zip(exact.blocks, universe.tower.stages):
        block = stage["block"]
        assert isinstance(block, StandardConvBlock1D)
        assert torch.equal(exact_block["ln"].norm.weight, block.norm.norm.weight)
        assert torch.equal(exact_block["ln"].norm.bias, block.norm.norm.bias)
        assert torch.equal(exact_block["conv"].weight, block.conv.weight)
        assert torch.equal(exact_block["conv"].bias, block.conv.bias)
    assert torch.equal(exact.pre_rnn_ln.norm.weight, universe.pre_rnn_norm.norm.weight)
    assert torch.equal(exact.gru.weight_ih_l0, universe.rnn.weight_ih_l0)
    assert torch.equal(exact.gru.weight_hh_l0, universe.rnn.weight_hh_l0)
    assert torch.equal(exact.gru.bias_ih_l0, universe.rnn.bias_ih_l0)
    assert torch.equal(exact.gru.bias_hh_l0, universe.rnn.bias_hh_l0)
    assert torch.equal(exact.bn1.weight, universe.head.norms[0].weight)
    assert torch.equal(exact.bn2.weight, universe.head.norms[1].weight)
    assert torch.equal(exact.fc1.weight, universe.head.linears[0].weight)
    assert torch.equal(exact.fc1.bias, universe.head.linears[0].bias)
    assert torch.equal(exact.fc2.weight, universe.head.linears[1].weight)
    assert torch.equal(exact.fc2.bias, universe.head.linears[1].bias)


def test_saluki_factory_matches_saluki_exact_initialization_and_forward():
    saluki_config = ConvolutionUniverseConfig.saluki()
    assert saluki_config.rnn_type == "gru"
    assert saluki_config.rnn_reverse is True
    assert saluki_config.stem_padding == "valid"
    assert saluki_config.head_norm_position == "pre"

    torch.manual_seed(17)
    exact = SalukiExact(**SalukiExactConfig().to_kwargs())
    torch.manual_seed(17)
    universe = ConvolutionUniverse(**saluki_config.to_kwargs())
    _assert_saluki_parameters_match(exact, universe)

    x = torch.randn(2, 6, 1024)
    exact.eval()
    universe.eval()
    assert torch.allclose(exact.forward_features(x), universe.forward_features(x), atol=1e-6)
    assert torch.allclose(exact(x), universe(x), atol=1e-6)
    assert torch.allclose(exact(x.transpose(1, 2)), universe(x.transpose(1, 2)), atol=1e-6)


def test_saluki_factory_matches_seeded_training_augmentation_and_dropout():
    torch.manual_seed(23)
    exact = SalukiExact(**SalukiExactConfig().to_kwargs()).train()
    torch.manual_seed(23)
    universe = ConvolutionUniverse(**ConvolutionUniverseConfig.saluki().to_kwargs()).train()
    x = torch.randn(4, 6, 1024)

    torch.manual_seed(99)
    expected = exact(x)
    torch.manual_seed(99)
    observed = universe(x)
    assert torch.allclose(expected, observed, atol=1e-6)


def test_saluki_ablations_use_ordinary_model_parameters():
    params = ConvolutionUniverseConfig.saluki().to_kwargs()
    params.update(
        {
            "augment_shift": 0,
            "pooling": "average",
            "head_norm": "layer",
            "rnn_bidirectional": True,
            "head_hidden": [32, 16],
        }
    )
    model = ConvolutionUniverse(**params).eval()
    assert model.rnn.bidirectional
    assert isinstance(model.head.norms[0], torch.nn.LayerNorm)
    assert model(torch.randn(2, 6, 1024)).shape == (2,)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"num_layers": 2, "dilations": [1]}, "exactly num_layers=2"),
        ({"stem_kernel_size": 4, "stem_padding": "same"}, "stem_kernel_size must be odd"),
        ({"kernel_size": 4, "padding": "same"}, "kernel_size must be odd"),
        ({"padding": "valid", "use_residual": True}, "requires padding='same'"),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"block_type": "attention"}, "block_type must be one of"),
    ],
)
def test_invalid_configs_raise_clear_errors(params, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ConvolutionUniverse(**params)


def test_short_valid_sequence_fails_before_pytorch_convolution_error():
    model = ConvolutionUniverse(**ConvolutionUniverseConfig.saluki().to_kwargs()).eval()
    with pytest.raises(ValueError, match="sequence length collapses"):
        model(torch.randn(2, 6, 16))
