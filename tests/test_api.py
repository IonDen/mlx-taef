"""Tests for the user-facing Taef family API."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef import TAEF1, TAEF2, TAESD, TAESDXL, Taef
from mlx_taef.variants import (
    TAEF1_CONFIG,
    TAEF2_CONFIG,
    TAESD_CONFIG,
    TAESDXL_CONFIG,
    TaesdVariantConfig,
)

CONVERTED_DIR = Path(__file__).parent / "converted"
REF_DIR = Path(__file__).parent / "reference"

# Primary decode parity tolerance. Pixel values are in [0, 1] and pass through
# zero, so a relative tolerance is undefined near zero (measured maxrel explodes
# to ~3.0 at near-black pixels) — we gate on absolute tolerance only. Measured
# worst maxabs across all 4 variants x 5 fixtures is ~1.1e-5 (MLX-Metal fp32 vs
# the PyTorch fp32 reference); 1e-4 clears that ~9x for cross-hardware fp32
# reduction-order noise while still failing a +0.05 brightness shift by ~500x.
# See mlx-taef-0001: a cosine-only gate (cos > 0.999) is DC-dominated on
# mid-gray images and accepts brightness/contrast/noise regressions.
DECODE_ATOL = 1e-4


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def test_taef2_loads_from_local_path() -> None:
    taef2 = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    # 32-channel FLUX.2 latent must reach the decoder's first conv (layers[1]).
    assert taef2.decoder.layers[1].weight.shape[-1] == 32


def test_taef2_decode_produces_correct_shape() -> None:
    taef2 = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    latent = mx.zeros((1, 64, 64, 32))
    img = taef2.decode(latent)
    assert img.shape == (1, 512, 512, 3)


def test_taef2_decode_output_clamped_to_zero_one() -> None:
    taef2 = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    latent = mx.random.normal((1, 64, 64, 32), scale=5.0)
    img = np.array(taef2.decode(latent))
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_taef2_decode_image_returns_uint8() -> None:
    taef2 = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    latent = mx.zeros((1, 64, 64, 32))
    img = taef2.decode_image(latent)
    assert img.dtype == mx.uint8
    assert img.shape == (1, 512, 512, 3)


def test_taef2_dtype_param_propagates() -> None:
    taef2 = TAEF2.from_pretrained_local(
        CONVERTED_DIR / "taef2_decoder.safetensors", dtype=mx.float16
    )
    # First conv after Clamp is decoder.layers[1].
    sample_param = taef2.decoder.layers[1].weight
    assert sample_param.dtype == mx.float16


def test_taef2_scale_unscale_latents_roundtrip() -> None:
    taef2 = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    raw = mx.random.normal((1, 8, 8, 32))
    # Use values within [-magnitude, magnitude] so clip doesn't lose info
    raw = mx.clip(raw, -2.9, 2.9)
    scaled = taef2.scale_latents(raw)
    unscaled = taef2.unscale_latents(scaled)
    assert np.allclose(np.array(unscaled), np.array(raw), atol=1e-5)


@pytest.mark.parametrize(
    ("variant_name", "cls", "config"),
    [
        ("taesd", TAESD, TAESD_CONFIG),
        ("taesdxl", TAESDXL, TAESDXL_CONFIG),
        ("taef1", TAEF1, TAEF1_CONFIG),
        ("taef2", TAEF2, TAEF2_CONFIG),
    ],
)
def test_variant_decode_against_committed_reference(
    variant_name: str,
    cls: type[Taef],
    config: TaesdVariantConfig,
) -> None:
    """Tier 2 parity for each variant: absolute pixel tolerance on 5 fixtures."""
    weights_path = CONVERTED_DIR / f"{variant_name}_decoder.safetensors"
    model = cls.from_pretrained_local(weights_path)
    for i in range(5):
        latent = mx.load(str(REF_DIR / f"{variant_name}_latent_{i:03d}.safetensors"))["latent"]
        ref = np.array(
            mx.load(str(REF_DIR / f"{variant_name}_decoded_{i:03d}.safetensors"))["image"]
        )
        out = np.array(model.decode(latent))
        # Absolute pixel tolerance is the gate (see DECODE_ATOL). Cosine
        # similarity is reported on failure as a secondary, scale-insensitive
        # diagnostic, not as the pass/fail criterion.
        np.testing.assert_allclose(
            out,
            ref,
            atol=DECODE_ATOL,
            rtol=0,
            err_msg=f"{variant_name} fixture {i}: cos_sim={_cosine_sim(out, ref):.6f}",
        )


def test_decode_parity_gate_rejects_brightness_shift() -> None:
    """The decode gate must reject a +0.05 brightness shift (mlx-taef-0001).

    Regression guard for the gate's strength: a +0.05 shift produces cosine
    similarity ~0.9996 (it sailed through the old `cos > 0.999` gate) but a
    maxabs of 0.05 — 500x the DECODE_ATOL=1e-4 bound. If someone later loosens
    the tolerance toward 0.05, this test catches it before a real brightness /
    scale / dropped-bias regression can slip past parity.
    """
    model = TAEF2.from_pretrained_local(CONVERTED_DIR / "taef2_decoder.safetensors")
    latent = mx.load(str(REF_DIR / "taef2_latent_000.safetensors"))["latent"]
    ref = np.array(mx.load(str(REF_DIR / "taef2_decoded_000.safetensors"))["image"])
    out = np.array(model.decode(latent))

    # The true output clears the gate ...
    np.testing.assert_allclose(out, ref, atol=DECODE_ATOL, rtol=0)
    # ... but a +0.05-shifted reference must not.
    shifted_ref = np.clip(ref + 0.05, 0.0, 1.0)
    assert _cosine_sim(out, shifted_ref) > 0.999  # old gate would have passed this
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out, shifted_ref, atol=DECODE_ATOL, rtol=0)
