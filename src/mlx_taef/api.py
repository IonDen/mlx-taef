"""User-facing Taef family API: load weights and decode/encode latents.

Tensor layout: all public methods use NHWC mx.array.
Value space: decode() outputs [0, 1] float; encode() expects [0, 1] float.
"""

import logging
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlx_taef.errors import TaefError
from mlx_taef.kernels import KERNELS, MIDBLOCK_GN, ModelKernel
from mlx_taef.kernels._arch import build_arch

logger = logging.getLogger(__name__)


class Taef(nn.Module):  # type: ignore[misc,name-defined]
    """Base class for TAESD-family models. Subclasses set `_kernel`."""

    _kernel: ModelKernel
    _decoder_loaded: bool
    _encoder_loaded: bool

    def __init__(self) -> None:
        """Build decoder + encoder from `self._kernel`."""
        super().__init__()
        self._decoder_loaded = False
        self._encoder_loaded = False
        ch = self._kernel.latent.channels
        mbgn = MIDBLOCK_GN.get(self._kernel.name, False)
        self.decoder = build_arch(
            self._kernel.arch.name, role="decoder", latent_channels=ch, midblock_gn=mbgn
        )
        self.encoder = build_arch(
            self._kernel.arch.name, role="encoder", latent_channels=ch, midblock_gn=mbgn
        )

    @classmethod
    def from_kernel(
        cls,
        kernel: ModelKernel,
        *,
        decoder_path: Path | str,
        encoder_path: Path | str | None = None,
        dtype: mx.Dtype = mx.float32,
    ) -> "Taef":
        """Build an instance bound to `kernel` and load converted MLX weights from disk."""

        class _Bound(cls):  # type: ignore[valid-type, misc]
            _kernel = kernel

        return cast(
            "Taef",
            _Bound.from_pretrained_local(decoder_path, encoder_path=encoder_path, dtype=dtype),
        )

    @classmethod
    def from_pretrained_local(
        cls,
        decoder_path: Path | str,
        encoder_path: Path | str | None = None,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> "Taef":
        """Instantiate from already-converted MLX safetensors on disk."""
        instance = cls()
        d_weights = cast("dict[str, mx.array]", mx.load(str(decoder_path)))
        instance.decoder.load_weights(list(d_weights.items()), strict=True)
        instance._decoder_loaded = True
        if encoder_path is not None:
            e_weights = cast("dict[str, mx.array]", mx.load(str(encoder_path)))
            instance.encoder.load_weights(list(e_weights.items()), strict=True)
            instance._encoder_loaded = True
        if dtype is not mx.float32:
            instance.set_dtype(dtype)
        instance.eval()
        return instance

    @classmethod
    def from_pretrained(  # pragma: no cover
        cls,
        repo_id: str | None = None,
        *,
        dtype: mx.Dtype = mx.float32,
        include_encoder: bool = True,
    ) -> "Taef":
        """Auto-download weights from HF Hub, convert to MLX, and load."""
        from mlx_taef.download import get_or_convert

        kernel = cls._kernel
        if repo_id is not None and repo_id != kernel.source.repo:
            raise ValueError(
                f"repo_id mismatch: requested {repo_id!r} but kernel {kernel.name!r} "
                f"uses {kernel.source.repo!r}"
            )
        decoder_path = get_or_convert(kernel, role="decoder")
        encoder_path = get_or_convert(kernel, role="encoder") if include_encoder else None
        return cls.from_pretrained_local(decoder_path, encoder_path=encoder_path, dtype=dtype)

    def decode(self, latents: mx.array) -> mx.array:
        """Decode raw latents (NHWC) to image (NHWC, [0, 1] float)."""
        if not self._decoder_loaded:
            raise TaefError(
                "decode() called before decoder weights were loaded. Build the model "
                "with from_pretrained() or from_pretrained_local(decoder_path=...)."
            )
        return mx.clip(self.decoder(latents), 0.0, 1.0)

    def decode_image(self, latents: mx.array) -> mx.array:
        """Decode raw latents to a uint8 NHWC image suitable for PIL/PNG."""
        return (self.decode(latents) * 255.0).astype(mx.uint8)

    def encode(self, image: mx.array) -> mx.array:
        """Encode an NHWC RGB image (B,H,W,3) in [0,1] to a latent (B,H/8,W/8,channels)."""
        if not self._encoder_loaded:
            raise TaefError(
                "encode() called on a model loaded without an encoder. Load with "
                "include_encoder=True (from_pretrained) or pass encoder_path= to "
                "from_pretrained_local()."
            )
        return cast("mx.array", self.encoder(image))

    def scale_latents(self, raw: mx.array) -> mx.array:
        """Map raw latents to [0, 1] using the kernel's latent magnitude/shift."""
        ls = self._kernel.latent
        return mx.clip(raw / (2.0 * ls.magnitude) + ls.shift, 0.0, 1.0)

    def unscale_latents(self, scaled: mx.array) -> mx.array:
        """Inverse of scale_latents: [0, 1] -> raw."""
        ls = self._kernel.latent
        return (scaled - ls.shift) * (2.0 * ls.magnitude)


class TAESD(Taef):
    """TAESD for Stable Diffusion 1.x."""

    _kernel = KERNELS["taesd"]


class TAESDXL(Taef):
    """TAESD for SDXL."""

    _kernel = KERNELS["taesdxl"]


class TAEF1(Taef):
    """TAEF1 for FLUX.1."""

    _kernel = KERNELS["taef1"]


class TAEF2(Taef):
    """TAEF2 for FLUX.2 Klein."""

    _kernel = KERNELS["taef2"]


__all__ = ["TAEF1", "TAEF2", "TAESD", "TAESDXL", "Taef"]
