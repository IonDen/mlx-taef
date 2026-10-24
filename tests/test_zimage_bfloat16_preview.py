"""Z-Image live preview on the bfloat16 in-loop latent mflux hands the callback.

The Z-Image latent mflux creates is bfloat16, and was already in mflux 0.19.2 (mflux 0.22 also
runs the Z-Image transformer in bfloat16; its `--float32` flag restores the old stream): the
committed Z-Image latent, captured from a real run, is stored as BF16. The preview decoder
holds float32 weights.
Offline: the committed latent and the committed taef1 decoder (Z-Image reuses its weights).
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mflux")

from mlx_taef.integrations.mflux import LivePreviewCallback

_LATENT = Path(__file__).parent / "fixtures" / "showcase_latents" / "z_image_turbo.safetensors"
_DECODER = Path(__file__).parent / "converted" / "taef1_decoder.safetensors"


@pytest.fixture
def zimage_callback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LivePreviewCallback:
    from mlx_taef import ZImage

    real = ZImage.from_pretrained_local(_DECODER)
    monkeypatch.setattr(ZImage, "from_pretrained", classmethod(lambda cls, **kw: real))
    return LivePreviewCallback(variant="zimage", every=1, save_to=tmp_path / "p.png")


def _preview(callback: LivePreviewCallback, latent: mx.array) -> np.ndarray:
    callback.call_before_loop(seed=0, prompt="", latents=latent, config=None)
    callback.call_in_loop(t=0, seed=0, prompt="", latents=latent, config=None, time_steps=None)
    assert callback.last_preview_source == "latents", "the preview was not written"
    return np.array(Image.open(callback.saved_paths[-1])).astype(np.int16)


def test_a_bfloat16_latent_previews_exactly_as_the_float32_latent_of_the_same_values(
    zimage_callback: LivePreviewCallback,
) -> None:
    """A bfloat16 latent and its float32 upcast carry the same numbers, so their previews must
    match pixel for pixel: the preview adds no rounding of its own to what the bfloat16 stream
    already carries. Without a cast at the unpack the decoder's first ops run in bfloat16 and
    the two previews differ by up to 6/255 (mean 0.12/255) on this latent. Catches: both
    float32 casts dropped (callback and unpack), or the preview refusing a bfloat16 latent."""
    bf16 = mx.load(str(_LATENT))["latent"]
    assert bf16.dtype == mx.bfloat16, bf16.dtype
    assert bf16.shape[:2] == (16, 1), bf16.shape

    from_bf16 = _preview(zimage_callback, bf16)
    from_fp32 = _preview(zimage_callback, bf16.astype(mx.float32))

    diff = np.abs(from_bf16 - from_fp32)
    assert diff.max() == 0, f"max {diff.max()}/255, mean {diff.mean():.4f}/255"


def test_the_zimage_unpack_returns_float32_for_a_bfloat16_latent() -> None:
    """The unpack itself hands TAEF1 float32, for callers that use the kernel binding outside
    the callback. Checks the dtype, not pixels. Catches: the float32 cast dropped from
    `unpack_zimage_latent`."""
    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.zimage import unpack_zimage_latent

    latent = mx.zeros((16, 1, 4, 6), dtype=mx.bfloat16)

    out = unpack_zimage_latent(latent, UnpackContext(latent_height=0, latent_width=0))

    assert out.dtype == mx.float32
    assert out.shape == (1, 4, 6, 16)
