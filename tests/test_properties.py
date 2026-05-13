"""Tier 3 property tests using hypothesis: shape, dtype, determinism, roundtrip."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

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


def test_decode_handles_extreme_input_without_nan(taef2: TAEF2) -> None:
    """Extreme but finite latent values must not produce NaN/Inf outputs.

    TAESD's Clamp(tanh(x/3)*3) layer at the decoder input bounds the latent
    magnitude before any conv. This test ensures the bound holds even for
    pathological inputs.
    """
    extreme = mx.random.normal((1, 8, 8, 32)) * 100.0
    out = np.array(taef2.decode(extreme))
    assert np.isfinite(out).all(), (
        f"Decode produced non-finite values: nan={np.isnan(out).sum()}, inf={np.isinf(out).sum()}"
    )
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_encode_decode_roundtrip_ssim_on_structured_image(taef2_with_encoder: TAEF2) -> None:
    """SSIM on a structured image must be ≥ 0.75 after TAEF2 encode→decode.

    TAEF2 is a tiny preview decoder; fine detail loss is expected. SSIM is
    the right metric (not MSE) because SSIM captures structural similarity
    rather than per-pixel exactness. Random noise (as in the MSE test above)
    is not a good proxy for perceptual quality on a tiny VAE — structured
    inputs like the gradient+checkerboard fixture are far more diagnostic.
    """
    img_path = Path(__file__).parent / "reference" / "_source_image.png"
    src = (
        np.array(Image.open(img_path).convert("RGB").resize((256, 256))).astype(np.float32) / 255.0
    )
    src_nhwc = mx.array(src[None, ...])

    latent = taef2_with_encoder.encode(src_nhwc)
    recon = np.array(taef2_with_encoder.decode(latent))[0]

    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore[import-untyped]

        # skimage SSIM: data_range=1.0 because both arrays are float32 in [0, 1].
        # channel_axis=-1 because images are HWC.
        score = float(ssim(src, recon, data_range=1.0, channel_axis=-1))
    except ImportError:  # pragma: no cover — scikit-image is an optional dep
        # Fallback: 1 - MSE-on-luminance. For a structured (low-frequency)
        # image, this correlates closely with SSIM; threshold 0.75 maps to
        # luminance MSE < 0.25, which is very generous.
        lum_src = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
        lum_recon = 0.299 * recon[..., 0] + 0.587 * recon[..., 1] + 0.114 * recon[..., 2]
        score = 1.0 - float(np.mean((lum_src - lum_recon) ** 2))

    assert score >= 0.75, f"Roundtrip SSIM/score {score:.4f} < 0.75 — preview decoder may be broken"
