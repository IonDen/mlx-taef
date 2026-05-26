"""mflux integration for mlx-taef.

Provides:
- `unpack_flux2_latent`: convert mflux's packed FLUX.2 latents to TAEF2-compatible NHWC.
- `LivePreviewCallback`: drop-in mflux callback that writes preview PNGs every N steps.

Install with: `pip install "mlx-taef[mflux]"`.
"""

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

try:
    from mflux.callbacks.callback import (  # type: ignore[import-untyped,import-not-found]
        InLoopCallback,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mflux is required for mlx_taef.integrations.mflux. "
        "Install with: pip install 'mlx-taef[mflux]'"
    ) from e

from mlx_taef.api import TAEF1, TAEF2, Taef

logger = logging.getLogger(__name__)


def unpack_flux2_latent(
    packed: mx.array,
    *,
    latent_height: int,
    latent_width: int,
    bn_mean: mx.array | None = None,
    bn_var: mx.array | None = None,
    bn_eps: float = 1e-4,
) -> mx.array:
    """Unpack mflux's packed FLUX.2 latent into NHWC `(B, lH*2, lW*2, 32)` for TAEF2.

    Mirrors mflux's `Flux2VAE.decode_packed_latents` preprocessing:
    1. BN denormalize: latents = packed * sqrt(var + eps) + mean  (128-channel stats)
    2. Unpatchify: (B, 128, lH, lW) -> (B, 32, lH*2, lW*2)
    3. NCHW -> NHWC transpose.

    Without `bn_mean`/`bn_var`, the helper assumes identity BN (correct shape,
    slight value offset). Preview structure will still be visible but colors
    may shift. For exact preview, pass the BN stats from the loaded Flux2VAE.

    Args:
        packed: in-loop latent from mflux, shape (B, lH*lW, 128).
        latent_height: latent spatial height (image_height // 16).
        latent_width: latent spatial width.
        bn_mean: optional BN running_mean for exact value recovery (128 elements).
        bn_var: optional BN running_var for exact value recovery (128 elements).
        bn_eps: BN epsilon. Matches mflux's Flux2BatchNormStats default of 1e-4,
            verified at mflux 0.17.5.

    Returns:
        NHWC tensor of shape (B, latent_height*2, latent_width*2, 32) — ready
        for `TAEF2.decode()`.

    See `notes/mflux-latent-layout.md` for the analysis behind this transform.
    """
    b, _, c = packed.shape
    if c != 128:
        raise ValueError(f"Expected 128-channel packed latent, got {c}")

    # Step 1: reshape and transpose to NCHW: (B, lH*lW, 128) -> (B, lH, lW, 128) -> (B, 128, lH, lW)
    latents = packed.reshape(b, latent_height, latent_width, c).transpose(0, 3, 1, 2)

    # Step 2: BN denormalize
    if bn_mean is not None and bn_var is not None:
        mean = bn_mean.reshape(1, -1, 1, 1)
        std = mx.sqrt(bn_var.reshape(1, -1, 1, 1) + bn_eps)
        latents = latents * std + mean
    # else: identity BN (mean=0, var=1)

    # Step 3: Unpatchify (B, 128, lH, lW) -> (B, 32, lH*2, lW*2)
    batch, _, h, w = latents.shape
    latents = latents.reshape(batch, 32, 2, 2, h, w)
    latents = latents.transpose(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch, 32, h * 2, w * 2)

    # Step 4: NCHW -> NHWC for TAEF2
    return latents.transpose(0, 2, 3, 1)


class LivePreviewCallback(InLoopCallback):  # type: ignore[misc]
    """mflux callback that writes a low-quality preview PNG every N iterations.

    Drops into `mflux.Flux2Klein.generate_image(callbacks=[...])`. On the
    iteration indices that match `every`, unpacks the in-flight latent, runs
    TAEF2 decode (via `TAEF1` for FLUX.1), and writes a PIL PNG to disk.

    Args:
        flux: optional reference to the mflux model instance the callback will be
            registered on. Typed as `object` to keep this module import-clean of
            mflux; intended to be a `Flux2Klein` instance when `auto_bn=True` and
            `variant="taef2"`. Required for auto-bn extraction (the mflux callback
            contract does not pass the flux instance at fire time).
        variant: 'taef1' (for FLUX.1 latents) or 'taef2' (for FLUX.2 Klein).
        every: emit a preview every Nth iteration. Default 5.
        save_to: filesystem path to write previews. Overwritten each emission.
        latent_height: latent spatial height; for 512x512 image with FLUX.2 Klein, this is 32.
        latent_width: latent spatial width.
        bn_mean: optional BN running_mean for TAEF2 (see `unpack_flux2_latent`).
        bn_var: optional BN running_var for TAEF2.
    """

    def __init__(
        self,
        *,
        flux: object | None = None,
        variant: str = "taef2",
        every: int = 5,
        save_to: str | Path = "preview.png",
        latent_height: int = 32,
        latent_width: int = 32,
        bn_mean: mx.array | None = None,
        bn_var: mx.array | None = None,
    ) -> None:
        """Initialise LivePreviewCallback. See class docstring for argument descriptions."""
        if variant == "taef1":  # pragma: no cover
            self.model: Taef = TAEF1.from_pretrained(include_encoder=False)
        elif variant == "taef2":
            self.model = TAEF2.from_pretrained(include_encoder=False)
        else:  # pragma: no cover
            raise ValueError(f"variant must be 'taef1' or 'taef2', got {variant!r}")
        self.flux = flux
        self.every = every
        self.save_to = Path(save_to)
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self._iter = 0

    def call_in_loop(
        self,
        t: object,
        seed: object,
        prompt: object,
        latents: mx.array,
        config: object,
        time_steps: object,
    ) -> None:
        """Decode latent + save PNG every Nth iteration (counted by our own iter, not `t`)."""
        idx = self._iter
        self._iter += 1
        if idx % self.every != 0:
            return  # pragma: no cover

        if self.model._config.name == "taef2":
            unpacked = unpack_flux2_latent(
                latents,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                bn_mean=self.bn_mean,
                bn_var=self.bn_var,
            )
        else:  # pragma: no cover
            # TAEF1 / FLUX.1: latents come in NCHW (B, 16, H, W). Transpose to NHWC.
            unpacked = mx.transpose(latents, (0, 2, 3, 1)) if latents.shape[1] == 16 else latents

        img = self.model.decode_image(unpacked)
        self._save_png(img[0])

    def _save_png(self, img_nhwc_uint8: mx.array) -> None:
        from PIL import Image

        arr = np.array(img_nhwc_uint8)
        Image.fromarray(arr).save(self.save_to)


__all__ = ["LivePreviewCallback", "unpack_flux2_latent"]
