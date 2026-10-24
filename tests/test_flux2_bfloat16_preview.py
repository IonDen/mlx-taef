"""Live preview on the bfloat16 in-loop latents mflux hands the callback.

mflux builds the FLUX.2 Klein in-loop latent in the model's precision (`prepare_latents` casts
to `ModelConfig.precision`, bfloat16): the committed Klein latent, captured from a real run, is
stored as BF16. The preview decoders hold float32 weights. Offline: the committed latent and
the committed taef2 decoder; the dtype check fakes the decode and loads only committed weights.
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mflux")

from mlx_taef import TAEF1, TAEF2, Krea2, QwenImage, ZImage
from mlx_taef.integrations.mflux import LivePreviewCallback

_CONVERTED = Path(__file__).parent / "converted"
_LATENT = (
    Path(__file__).parent / "fixtures" / "showcase_latents" / "flux2_klein_base_4b.safetensors"
)


@dataclass(frozen=True)
class _Config:
    """The two fields of mflux's Config the callback reads to unpack a packed latent."""

    height: int
    width: int


def _offline(monkeypatch: pytest.MonkeyPatch, cls: type, weights: str) -> None:
    real = cls.from_pretrained_local(_CONVERTED / weights)
    monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, **kw: real))


def _preview(callback: LivePreviewCallback, latent: mx.array, config: _Config) -> np.ndarray:
    callback.call_before_loop(seed=0, prompt="", latents=latent, config=config)
    callback.call_in_loop(t=0, seed=0, prompt="", latents=latent, config=config, time_steps=None)
    assert callback.last_preview_source == "latents", "the preview was not written"
    return np.array(Image.open(callback.saved_paths[-1])).astype(np.int16)


def test_a_bfloat16_klein_latent_previews_exactly_as_its_float32_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bfloat16 latent and its float32 upcast carry the same numbers, so their previews must
    match pixel for pixel. Without the float32 cast in the callback, taef2's clamp and first
    convolutions run in bfloat16 and the two previews of this latent differ by up to 2/255
    (mean 0.0323/255). Catches: the central cast dropped from the preview emission."""
    _offline(monkeypatch, TAEF2, "taef2_decoder.safetensors")
    arrays = mx.load(str(_LATENT))
    bf16 = arrays["latent"]
    config = _Config(height=int(arrays["height"].item()), width=int(arrays["width"].item()))
    assert bf16.dtype == mx.bfloat16, bf16.dtype
    assert bf16.ndim == 3, bf16.shape
    assert bf16.shape[2] == 128, bf16.shape
    callback = LivePreviewCallback(variant="taef2", every=1, save_to=tmp_path / "p.png")

    from_bf16 = _preview(callback, bf16, config)
    from_fp32 = _preview(callback, bf16.astype(mx.float32), config)

    diff = np.abs(from_bf16 - from_fp32)
    assert diff.max() == 0, f"max {diff.max()}/255, mean {diff.mean():.4f}/255"


# Variant -> (API class, committed decoder weights, a bf16 in-loop latent of that model's
# layout for a 64x64 image). Packed latents read lh = lw = 64 // 16 = 4 from the config.
_LATENTS_BY_VARIANT: dict[str, tuple[type, str, tuple[int, ...]]] = {
    "taef1": (TAEF1, "taef1_decoder.safetensors", (1, 16, 64)),
    "taef2": (TAEF2, "taef2_decoder.safetensors", (1, 16, 128)),
    "zimage": (ZImage, "taef1_decoder.safetensors", (16, 1, 8, 8)),
    "qwen-image": (QwenImage, "qwen-image_decoder.safetensors", (1, 16, 64)),
    "krea2": (Krea2, "qwen-image_decoder.safetensors", (1, 16, 8, 8)),
}


@pytest.mark.parametrize("variant", list(_LATENTS_BY_VARIANT))
def test_every_variant_hands_its_decoder_a_float32_latent(
    variant: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whatever dtype mflux hands over, the decoder receives float32, the dtype of its weights.
    Reads the dtype at the decoder's input, so it does not depend on how large the bfloat16
    rounding is for a given latent. Catches: the float32 cast removed from the callback (every
    variant but zimage, whose unpack casts too, would decode a bfloat16 latent)."""
    cls, weights, shape = _LATENTS_BY_VARIANT[variant]
    _offline(monkeypatch, cls, weights)
    callback = LivePreviewCallback(variant=variant, every=1, save_to=tmp_path / "p.png")  # type: ignore[arg-type]
    seen: list[mx.Dtype] = []

    def _decode_image(latents: mx.array) -> mx.array:
        seen.append(latents.dtype)
        return mx.zeros((1, 8, 8, 3), dtype=mx.uint8)

    monkeypatch.setattr(callback.model, "decode_image", _decode_image)
    latent = mx.zeros(shape, dtype=mx.bfloat16)

    callback.call_in_loop(
        t=0, seed=0, prompt="", latents=latent, config=_Config(64, 64), time_steps=None
    )

    assert seen == [mx.float32]
