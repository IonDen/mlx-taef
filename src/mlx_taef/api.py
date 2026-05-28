"""User-facing Taef family API: load weights and decode/encode latents.

Tensor layout: all public methods use NHWC mx.array.
Value space: decode() outputs [0, 1] float; encode() expects [0, 1] float.
"""

import logging
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from mlx_taef.model import make_decoder, make_encoder
from mlx_taef.variants import (
    TAEF1_CONFIG,
    TAEF2_CONFIG,
    TAESD_CONFIG,
    TAESDXL_CONFIG,
    TaesdVariantConfig,
)

logger = logging.getLogger(__name__)


class Taef(nn.Module):  # type: ignore[misc,name-defined]
    """Base class for TAESD-family models.

    Subclasses set the `_config` class variable to a `TaesdVariantConfig`.

    Args:
        None — instantiated via `from_pretrained_local(path)`.

    Tensor layout:
        All public methods use NHWC `mx.array`:
        - latent shape: (B, H, W, latent_channels)
        - image shape: (B, H*8, W*8, 3)

    Value space:
        - `decode()` returns float in [0, 1] (clipped on output).
        - `decode_image()` returns uint8 in [0, 255].
        - Latents are RAW (post-Clamp scale), values roughly in [-3, 3].
        - `scale_latents` / `unscale_latents` convert between raw and [0, 1].
    """

    _config: TaesdVariantConfig

    def __init__(self) -> None:
        """Initialise decoder and encoder from the subclass `_config`."""
        super().__init__()
        self.decoder = make_decoder(self._config)
        self.encoder = make_encoder(self._config)

    @classmethod
    def from_pretrained_local(
        cls,
        decoder_path: Path | str,
        encoder_path: Path | str | None = None,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> "Taef":
        """Instantiate from already-converted MLX safetensors on disk.

        Args:
            decoder_path: path to converted MLX decoder safetensors.
            encoder_path: optional path to converted MLX encoder safetensors (Phase 2).
            dtype: target dtype for model weights. Default fp32 to keep parity tests sharp.

        Returns:
            A loaded `Taef` instance ready for `decode()`.
        """
        instance = cls()
        # Load each submodule with strict=True so a dropped or wrong-shaped weight
        # raises instead of leaving a parameter at random init (silently-wrong
        # image). Loading per-submodule — rather than the whole instance — keeps
        # decoder-only loading valid: the encoder simply stays at init when no
        # encoder_path is given, instead of tripping a top-level strict check on
        # the (legitimately absent) encoder parameters.
        d_weights = cast("dict[str, mx.array]", mx.load(str(decoder_path)))
        instance.decoder.load_weights(list(d_weights.items()), strict=True)
        if encoder_path is not None:
            e_weights = cast("dict[str, mx.array]", mx.load(str(encoder_path)))
            instance.encoder.load_weights(list(e_weights.items()), strict=True)
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
        """Auto-download weights from HF Hub, convert to MLX, and load.

        On first call, downloads upstream weights from `cls._config.hf_repo` and
        caches converted output at ~/.cache/mlx-taef/. Subsequent calls return
        instantly from cache.

        Args:
            repo_id: optional HF repo override. If set, must match `cls._config.hf_repo`.
            dtype: target dtype for weights. Default fp32 to keep parity strict.
            include_encoder: whether to also load the encoder side (default True).

        Returns:
            A loaded `Taef` instance.
        """
        from mlx_taef.download import get_or_convert

        config = cls._config
        if repo_id is not None and repo_id != config.hf_repo:
            raise ValueError(
                f"repo_id mismatch: requested {repo_id!r} but variant {config.name!r} "
                f"uses {config.hf_repo!r}"
            )
        decoder_path = get_or_convert(config, role="decoder")
        encoder_path = get_or_convert(config, role="encoder") if include_encoder else None
        return cls.from_pretrained_local(decoder_path, encoder_path=encoder_path, dtype=dtype)

    def decode(self, latents: mx.array) -> mx.array:
        """Decode raw latents (NHWC) to image (NHWC, [0, 1] float).

        Args:
            latents: NHWC array with shape (B, H, W, latent_channels).

        Returns:
            NHWC float array with shape (B, H*8, W*8, 3), values in [0, 1].
        """
        out = self.decoder(latents)
        return mx.clip(out, 0.0, 1.0)

    def decode_image(self, latents: mx.array) -> mx.array:
        """Decode raw latents to a uint8 NHWC image suitable for PIL/PNG.

        Args:
            latents: NHWC array with shape (B, H, W, latent_channels).

        Returns:
            NHWC uint8 array with shape (B, H*8, W*8, 3), values in [0, 255].
        """
        img_float = self.decode(latents)
        return (img_float * 255.0).astype(mx.uint8)

    def encode(self, image: mx.array) -> mx.array:
        """Encode an NHWC RGB image (B, H, W, 3) in [0, 1] to a latent (B, H/8, W/8, latent_channels).

        Args:
            image: NHWC float array with shape (B, H, W, 3), values in [0, 1].

        Returns:
            NHWC latent array with shape (B, H/8, W/8, latent_channels).
        """
        return cast("mx.array", self.encoder(image))

    def scale_latents(self, raw: mx.array) -> mx.array:
        """Map raw latents to [0, 1] using config.latent_magnitude/shift.

        Args:
            raw: raw latent array, values roughly in [-3, 3].

        Returns:
            Scaled latent array clipped to [0, 1].
        """
        c = self._config
        return mx.clip(raw / (2.0 * c.latent_magnitude) + c.latent_shift, 0.0, 1.0)

    def unscale_latents(self, scaled: mx.array) -> mx.array:
        """Inverse of scale_latents: [0, 1] -> raw.

        Args:
            scaled: latent array in [0, 1].

        Returns:
            Raw latent array.
        """
        c = self._config
        return (scaled - c.latent_shift) * (2.0 * c.latent_magnitude)


class TAESD(Taef):
    """TAESD for Stable Diffusion 1.x."""

    _config = TAESD_CONFIG


class TAESDXL(Taef):
    """TAESD for SDXL."""

    _config = TAESDXL_CONFIG


class TAEF1(Taef):
    """TAEF1 for FLUX.1."""

    _config = TAEF1_CONFIG


class TAEF2(Taef):
    """TAEF2 for FLUX.2 Klein."""

    _config = TAEF2_CONFIG


__all__ = ["TAEF1", "TAEF2", "TAESD", "TAESDXL", "Taef"]
