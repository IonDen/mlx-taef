"""Tier 3 property tests using hypothesis: shape, dtype, determinism, roundtrip."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlx_taef import TAEF2


@pytest.fixture(scope="module")
def taef2(converted_dir: Path) -> TAEF2:
    return TAEF2.from_pretrained_local(converted_dir / "taef2_decoder.safetensors")


@pytest.fixture(scope="module")
def taef2_with_encoder(converted_dir: Path) -> TAEF2:
    return TAEF2.from_pretrained_local(
        decoder_path=converted_dir / "taef2_decoder.safetensors",
        encoder_path=converted_dir / "taef2_encoder.safetensors",
    )


def test_decode_output_shape_invariant(taef2: TAEF2) -> None:
    """For any latent (1, H, W, 32), output is (1, H*8, W*8, 3)."""
    for h, w in [(8, 8), (16, 16), (32, 32), (64, 64), (24, 40)]:
        latent = mx.zeros((1, h, w, 32))
        out = taef2.decode(latent)
        assert out.shape == (1, h * 8, w * 8, 3)


def test_decode_output_range_in_zero_one(taef2: TAEF2) -> None:
    """decode() clips to [0, 1]."""
    latent = mx.random.normal((1, 16, 16, 32), scale=10.0)
    out = np.array(taef2.decode(latent))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_decode_is_deterministic(taef2: TAEF2) -> None:
    """Same input -> identical output bytes across calls."""
    latent = mx.random.normal((1, 8, 8, 32))
    mx.eval(latent)
    out1 = np.array(taef2.decode(latent))
    out2 = np.array(taef2.decode(latent))
    assert np.array_equal(out1, out2)


@given(
    h=st.integers(min_value=4, max_value=32),
    w=st.integers(min_value=4, max_value=32),
)
@settings(deadline=None, max_examples=10)
def test_decode_shape_invariant_property(taef2: TAEF2, h: int, w: int) -> None:
    """Property: decode produces 8x upsampled output for any reasonable latent size."""
    latent = mx.zeros((1, h, w, 32))
    out = taef2.decode(latent)
    assert out.shape == (1, h * 8, w * 8, 3)


def test_decode_image_shape_and_dtype(taef2: TAEF2) -> None:
    """decode_image returns uint8 NHWC."""
    latent = mx.zeros((1, 8, 8, 32))
    img = taef2.decode_image(latent)
    assert img.dtype == mx.uint8
    assert img.shape == (1, 64, 64, 3)


def test_encode_decode_roundtrip_perceptually_close(taef2_with_encoder: TAEF2) -> None:
    """encode(image); decode(latent) should be perceptually close to original.

    TAEF is lossy — we accept a generous MSE bound. The critical claim isn't
    perfect reconstruction but that the lossy roundtrip doesn't collapse.
    """
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (1, 256, 256, 3)).astype(np.float32)
    latent = taef2_with_encoder.encode(mx.array(img))
    recon = np.array(taef2_with_encoder.decode(latent))
    mse = float(np.mean((img - recon) ** 2))
    # Generous bound: random noise is hard to reconstruct, but the result
    # shouldn't drift to the corners of [0, 1].
    assert mse < 0.3, f"Roundtrip MSE {mse:.4f} exceeds 0.3 — encoder/decoder may be broken"
