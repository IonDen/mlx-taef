"""LivePreviewCallback variant dispatch is registry-driven and includes qwen-image."""

import pytest


def test_variant_registry_includes_qwen_image():
    pytest.importorskip("mflux")
    from mlx_taef.api import QwenImage
    from mlx_taef.integrations.mflux import _VARIANT_CLASSES

    assert _VARIANT_CLASSES["qwen-image"] is QwenImage


def test_variant_registry_includes_krea2():
    pytest.importorskip("mflux")
    from mlx_taef.api import Krea2
    from mlx_taef.integrations.mflux import _VARIANT_CLASSES

    assert _VARIANT_CLASSES["krea2"] is Krea2


def test_livepreview_rejects_unknown_variant():
    pytest.importorskip("mflux")
    from mlx_taef.integrations.mflux import LivePreviewCallback

    with pytest.raises(ValueError, match="variant must be one of"):
        LivePreviewCallback(variant="not-a-kernel")


def test_unpack_qwen_latent_produces_taehv_nhwc_shape():
    """unpack_qwen_latent maps packed (B, lh*lw, 64) -> NHWC (B, lh*2, lw*2, 16). Offline-safe
    (pure MLX, no weights). Non-square dims so a height/width transposition reds the shape."""
    import mlx.core as mx

    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.qwen import unpack_qwen_latent

    lh, lw = 4, 6
    out = unpack_qwen_latent(
        mx.zeros((1, lh * lw, 64)), UnpackContext(latent_height=lh, latent_width=lw)
    )
    assert out.shape == (1, lh * 2, lw * 2, 16)


def test_unpack_qwen_latent_rejects_non_64_channels():
    import mlx.core as mx

    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.qwen import unpack_qwen_latent

    with pytest.raises(ValueError, match="64-channel"):
        unpack_qwen_latent(mx.zeros((1, 16, 16)), UnpackContext(latent_height=4, latent_width=4))
