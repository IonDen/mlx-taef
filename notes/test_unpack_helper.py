"""Synthetic-shape tests for unpack_flux2_latent.

Cannot test against real mflux output without running a real generation
(which needs multi-GB model download). These tests verify shape correctness
and that the unpacked output is decodable by our TAEF2 implementation.

Manual post-deployment validation:
  1. Run a real mflux FLUX.2 Klein generation with an InLoopCallback that
     captures `latents` at step 0.
  2. Call unpack_flux2_latent(captured_latents, ...) and compare the
     resulting image with the reference mflux stepwise output.
  3. For exact fidelity, also pass bn_mean and bn_var from
     Flux2VAE.bn.running_mean / running_var.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from unpack_helper import unpack_flux2_latent

from mlx_taef import TAEF2

CONVERTED = Path(__file__).parent.parent / "tests" / "converted" / "taef2_decoder.safetensors"


# ---------------------------------------------------------------------------
# Shape correctness tests (no model needed)
# ---------------------------------------------------------------------------


def test_unpack_basic_shape_512x512() -> None:
    """512×512 image: latent_h=latent_w=32, packed (1, 1024, 128) -> (1, 64, 64, 32)."""
    latent_h, latent_w = 32, 32  # 512 // 16
    packed = mx.random.normal((1, latent_h * latent_w, 128))
    out = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert out.shape == (1, 64, 64, 32), f"Got {out.shape}"


def test_unpack_basic_shape_1024x1024() -> None:
    """1024×1024 image: latent_h=latent_w=64, packed (1, 4096, 128) -> (1, 128, 128, 32)."""
    latent_h, latent_w = 64, 64  # 1024 // 16
    packed = mx.random.normal((1, latent_h * latent_w, 128))
    out = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert out.shape == (1, 128, 128, 32), f"Got {out.shape}"


def test_unpack_ndim() -> None:
    """Output must be 4-D NHWC."""
    packed = mx.random.normal((1, 32 * 32, 128))
    out = unpack_flux2_latent(packed, latent_height=32, latent_width=32)
    assert out.ndim == 4


def test_unpack_32_channels() -> None:
    """Last dimension must be 32 (TAEF2 latent channels)."""
    packed = mx.random.normal((1, 32 * 32, 128))
    out = unpack_flux2_latent(packed, latent_height=32, latent_width=32)
    assert out.shape[-1] == 32, f"Expected 32 channels (TAEF2), got {out.shape}"


def test_unpack_batch_size_2() -> None:
    """Batch size > 1 should work."""
    latent_h, latent_w = 32, 32
    packed = mx.random.normal((2, latent_h * latent_w, 128))
    out = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert out.shape == (2, 64, 64, 32), f"Got {out.shape}"


def test_unpack_nonsquare() -> None:
    """Non-square latent (e.g. 512x768 image -> latent 32x48)."""
    latent_h, latent_w = 32, 48  # 512//16=32, 768//16=48
    packed = mx.random.normal((1, latent_h * latent_w, 128))
    out = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert out.shape == (1, 64, 96, 32), f"Got {out.shape}"


def test_unpack_with_bn_stats() -> None:
    """BN de-normalization path: passing bn_mean/bn_var should not crash."""
    latent_h, latent_w = 32, 32
    packed = mx.random.normal((1, latent_h * latent_w, 128))
    bn_mean = mx.zeros((128,))
    bn_var = mx.ones((128,))
    out = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=bn_mean,
        bn_var=bn_var,
    )
    assert out.shape == (1, 64, 64, 32)


def test_unpack_identity_bn_matches_no_bn() -> None:
    """Identity BN (mean=0, var=1) gives near-identical result as no BN.

    With var=1 the BN std is sqrt(1 + eps) ≈ 1.00005, so values differ by
    at most ~5e-5 × |value|.  We use atol=1e-3 to account for float32 noise.
    """
    latent_h, latent_w = 32, 32
    key = mx.random.key(42)
    packed = mx.random.normal((1, latent_h * latent_w, 128), key=key)

    out_no_bn = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    out_id_bn = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=mx.zeros((128,)),
        bn_var=mx.ones((128,)),
    )
    mx.eval(out_no_bn, out_id_bn)
    # identity BN has std = sqrt(1 + bn_eps) ≈ 1.0001, so the results are very
    # close but not bit-for-bit identical; 1e-3 is safe for float32.
    assert mx.allclose(out_no_bn, out_id_bn, atol=1e-3).item()


def test_unpack_bad_seq_len_raises() -> None:
    """Wrong seq_len must raise ValueError."""
    with pytest.raises(ValueError, match="seq_len mismatch"):
        unpack_flux2_latent(
            mx.zeros((1, 999, 128)),
            latent_height=32,
            latent_width=32,
        )


def test_unpack_bad_channels_raises() -> None:
    """Wrong channel count must raise ValueError."""
    with pytest.raises(ValueError, match="128 channels"):
        unpack_flux2_latent(
            mx.zeros((1, 32 * 32, 64)),
            latent_height=32,
            latent_width=32,
        )


def test_unpack_bad_ndim_raises() -> None:
    """2-D input must raise ValueError."""
    with pytest.raises(ValueError, match="ndim=3"):
        unpack_flux2_latent(
            mx.zeros((1024, 128)),
            latent_height=32,
            latent_width=32,
        )


# ---------------------------------------------------------------------------
# End-to-end TAEF2 decodability test (requires converted weights)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CONVERTED.exists(), reason="taef2_decoder.safetensors not found")
def test_unpack_produces_taef2_compatible_shape() -> None:
    """unpack output must be feedable into TAEF2.decode without error."""
    latent_h, latent_w = 32, 32  # 512×512 image
    packed = mx.random.normal((1, latent_h * latent_w, 128))
    unpacked = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)

    assert unpacked.ndim == 4
    assert unpacked.shape[-1] == 32, f"Expected 32 channels (TAEF2), got {unpacked.shape}"

    taef2 = TAEF2.from_pretrained_local(CONVERTED)
    img = taef2.decode(unpacked)
    mx.eval(img)

    assert img.ndim == 4, f"Expected 4-D image, got ndim={img.ndim}"
    assert img.shape[-1] == 3, f"Expected 3 RGB channels, got {img.shape[-1]}"
    assert img.shape[1] == latent_h * 2 * 8, f"Unexpected height: {img.shape}"  # 512
    assert img.shape[2] == latent_w * 2 * 8, f"Unexpected width: {img.shape}"  # 512
