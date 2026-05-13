"""Tests for HF -> MLX weight conversion."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_taef.convert import convert_diffusers_to_sequential, convert_hf_decoder_to_mlx
from mlx_taef.model import make_decoder
from mlx_taef.variants import TAEF2_CONFIG


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
