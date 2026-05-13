"""Tests for HF -> MLX weight conversion."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.convert import (
    _build_mlx_state_dict,
    _flatten_module_param_shapes,
    _sequential_key_to_mlx,
    convert_diffusers_to_sequential,
    convert_hf_decoder_to_mlx,
    convert_hf_encoder_to_mlx,
)
from mlx_taef.model import make_decoder, make_encoder
from mlx_taef.variants import ALL_VARIANTS, TAEF2_CONFIG


def test_diffusers_key_mapper_decoder_gets_plus_one_offset():
    """Per upstream model card: decoder Diffusers keys get +1 index offset."""
    sd = {
        "decoder.layers.0.weight": "ignore",
        "decoder.layers.3.weight": "ignore",
        "encoder.layers.0.weight": "ignore",
    }
    decoder_remap = convert_diffusers_to_sequential(sd, role="decoder")
    # decoder: 0 -> 1, 3 -> 4
    assert "1.weight" in decoder_remap
    assert "4.weight" in decoder_remap


def test_diffusers_key_mapper_encoder_gets_no_offset():
    sd = {
        "encoder.layers.0.weight": "ignore",
        "encoder.layers.3.weight": "ignore",
    }
    encoder_remap = convert_diffusers_to_sequential(sd, role="encoder")
    # encoder: no offset
    assert "0.weight" in encoder_remap
    assert "3.weight" in encoder_remap


def test_diffusers_key_mapper_filters_other_role():
    sd = {
        "decoder.layers.0.weight": "kept",
        "encoder.layers.0.weight": "filtered",
    }
    decoder_only = convert_diffusers_to_sequential(sd, role="decoder")
    assert "1.weight" in decoder_only  # decoder 0+1=1
    assert "0.weight" not in decoder_only  # encoder skipped


def test_sequential_key_to_mlx_flat_key() -> None:
    # Flat key like "0.weight" becomes "layers.0.weight"
    assert _sequential_key_to_mlx("0.weight") == "layers.0.weight"


def test_sequential_key_to_mlx_nested_sequential() -> None:
    # Nested: "3.conv.0.weight" -> "layers.3.conv.layers.0.weight"
    out = _sequential_key_to_mlx("3.conv.0.weight")
    assert out.startswith("layers.")
    assert "conv" in out


def test_build_mlx_state_dict_transposes_4d_conv_weights() -> None:
    # NCHW conv weight (out=2, in=3, kH=3, kW=3)
    nchw = np.zeros((2, 3, 3, 3), dtype=np.float32)
    nchw[0, 0, 0, 0] = 1.0
    sd = {"1.weight": nchw}
    # Expected NHWC shape: (out=2, kH=3, kW=3, in=3) == (2, 3, 3, 3)
    expected = {"layers.1.weight": (2, 3, 3, 3)}
    out = _build_mlx_state_dict(sd, expected_shapes=expected)
    assert "layers.1.weight" in out
    assert tuple(out["layers.1.weight"].shape) == (2, 3, 3, 3)


def test_build_mlx_state_dict_skips_unmapped_keys() -> None:
    sd = {"some.extra.key": np.zeros((2, 2))}
    out = _build_mlx_state_dict(sd, expected_shapes={"layers.0.weight": (2, 2)})
    assert out == {}


def test_flatten_module_param_shapes_walks_nested_sequentials() -> None:
    from mlx_taef.variants import TAEF2_CONFIG

    shapes = _flatten_module_param_shapes(make_decoder(TAEF2_CONFIG))
    # First conv after Clamp at layers[1]: 32 -> 64, weight shape (64, 3, 3, 32) NHWC
    assert "layers.1.weight" in shapes
    assert shapes["layers.1.weight"] == (64, 3, 3, 32)


@pytest.mark.network
def test_taef2_conversion_produces_expected_keys(tmp_path: Path) -> None:
    """End-to-end: download taef2.safetensors and convert. Marked network because it hits HF."""
    out_path = tmp_path / "taef2_decoder.safetensors"
    convert_hf_decoder_to_mlx(out_path=out_path, config=TAEF2_CONFIG)
    weights = mx.load(str(out_path))
    decoder = make_decoder(TAEF2_CONFIG)
    expected_keys = set(_flatten_param_paths(decoder.parameters()))
    actual_keys = set(weights.keys())
    assert expected_keys == actual_keys, (
        f"Missing keys: {sorted(expected_keys - actual_keys)}\n"
        f"Extra keys: {sorted(actual_keys - expected_keys)}"
    )


@pytest.mark.network
def test_conv_weights_are_transposed_to_nhwc(tmp_path: Path) -> None:
    out_path = tmp_path / "taef2_decoder.safetensors"
    convert_hf_decoder_to_mlx(out_path=out_path, config=TAEF2_CONFIG)
    weights = mx.load(str(out_path))
    # First conv after Clamp is layers[1] in the Sequential: 32 in -> 64 out, 3x3 kernel
    assert "layers.1.weight" in weights
    # MLX NHWC shape: (out=64, kH=3, kW=3, in=32)
    assert weights["layers.1.weight"].shape == (64, 3, 3, 32)


@pytest.mark.network
@pytest.mark.parametrize("variant_name", ["taesd", "taesdxl", "taef1", "taef2"])
@pytest.mark.parametrize("role", ["decoder", "encoder"])
def test_fresh_conversion_keys_match_model(variant_name: str, role: str, tmp_path: Path) -> None:
    config = next(v for v in ALL_VARIANTS if v.name == variant_name)
    out_path = tmp_path / f"{variant_name}_{role}.safetensors"
    if role == "decoder":
        convert_hf_decoder_to_mlx(out_path=out_path, config=config)
        module = make_decoder(config)
    else:
        convert_hf_encoder_to_mlx(out_path=out_path, config=config)
        module = make_encoder(config)

    weights = mx.load(str(out_path))
    expected_keys = set(_flatten_param_paths(module.parameters()))
    actual_keys = set(weights.keys())
    assert expected_keys == actual_keys, (
        f"{variant_name} {role}: missing={sorted(expected_keys - actual_keys)} "
        f"extra={sorted(actual_keys - expected_keys)}"
    )


def _flatten_param_paths(params, prefix: str = ""):
    """Recursively walk a parameters() dict and yield dotted param paths."""
    keys = []
    if isinstance(params, dict):
        for k, v in params.items():
            keys.extend(_flatten_param_paths(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(params, list):
        for i, item in enumerate(params):
            keys.extend(_flatten_param_paths(item, f"{prefix}.{i}"))
    elif hasattr(params, "shape"):
        keys.append(prefix)
    return keys
