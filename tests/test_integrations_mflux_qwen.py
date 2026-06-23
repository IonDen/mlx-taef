"""LivePreviewCallback variant dispatch is registry-driven and includes qwen-image."""

import pytest


def test_variant_registry_includes_qwen_image():
    pytest.importorskip("mflux")
    from mlx_taef.api import QwenImage
    from mlx_taef.integrations.mflux import _VARIANT_CLASSES

    assert _VARIANT_CLASSES["qwen-image"] is QwenImage


def test_livepreview_rejects_unknown_variant():
    pytest.importorskip("mflux")
    from mlx_taef.integrations.mflux import LivePreviewCallback

    with pytest.raises(ValueError, match="variant must be one of"):
        LivePreviewCallback(variant="not-a-kernel")
