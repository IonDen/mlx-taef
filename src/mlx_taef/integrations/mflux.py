"""mflux integration for mlx-taef.

Provides:
- `unpack_flux2_latent`: convert mflux's packed FLUX.2 latents to TAEF2-compatible NHWC.
- `LivePreviewCallback`: drop-in mflux callback that writes preview PNGs every N steps.
  Supports FLUX.1 (``variant='taef1'``), FLUX.2 Klein (``variant='taef2'``), and Z-Image /
  Z-Image-Turbo (``variant='zimage'``, reuses TAEF1 weights).

Install with: `pip install "mlx-taef[mflux]"`.
"""

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_taef.errors import MfluxNotInstalledError

try:
    from mflux.callbacks.callback import (  # type: ignore[import-untyped,import-not-found]
        InLoopCallback,
    )
except ImportError as e:  # pragma: no cover
    raise MfluxNotInstalledError() from e

from mlx_taef.api import TAEF1, TAEF2, Taef, ZImage

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
    """Back-compat keyword API for the FLUX.2 unpack; delegates to the kernel unpack.

    The canonical implementation now lives in `mlx_taef.kernels.flux.unpack_flux2_latent`
    with the `(latent, UnpackContext)` signature; this wrapper preserves the original
    keyword call shape for existing callers.
    """
    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.flux import unpack_flux2_latent as _kernel_unpack

    ctx = UnpackContext(
        latent_height=latent_height,
        latent_width=latent_width,
        bn_mean=bn_mean,
        bn_var=bn_var,
        bn_eps=bn_eps,
    )
    return _kernel_unpack(packed, ctx)


def _try_extract_bn(
    flux: object,
) -> tuple[mx.array | None, mx.array | None, float | None]:
    """Best-effort extraction of FLUX.2 VAE BN stats.

    Returns (running_mean, running_var, eps) on success, (None, None, None) on any
    failure (missing attribute, wrong shape, exception). Never raises.
    """
    try:
        vae = getattr(flux, "vae", None)
        if vae is None:
            return (None, None, None)
        bn = getattr(vae, "bn", None)
        if bn is None:
            return (None, None, None)
        mean = getattr(bn, "running_mean", None)
        var = getattr(bn, "running_var", None)
        if mean is None or var is None:
            return (None, None, None)
        eps = getattr(bn, "eps", None)
        return (mean, var, eps)
    except Exception:
        return (None, None, None)


def _resolve_latent_dims(
    latent_height: int | None,
    latent_width: int | None,
    config: object,
    downscale: int,
) -> tuple[int, int]:
    """Return the (latent_height, latent_width) to unpack the in-loop latent with.

    Explicit dims (both provided) override and are returned unchanged. Otherwise they are
    derived from the mflux Config's image dimensions: ``lh = config.height // downscale``,
    ``lw = config.width // downscale``. mflux forces ``config.height/width`` to multiples of
    16, and FLUX in-loop latents are packed so ``lh = image // 16`` (see
    mflux/models/common/config/config.py and kernels/flux.py). Raises ``ValueError`` when
    dims are absent and the config does not expose ``.height``/``.width``.

    Precondition: callers pass both dims or neither — the LivePreviewCallback constructor
    rejects exactly-one upstream, so the partial case never reaches here. Only bindings whose
    in-loop latent is packed (downscale not None) reach this helper; the unpacked Z-Image path
    skips it. Invariant: mflux rounds config H/W to multiples of 16 unconditionally
    (config.py: ``16 * (height // 16)``), so a packed binding whose downscale is not itself a
    factor of 16 would desync from mflux's rounding — today every packed binding is 16.
    """
    if latent_height is not None and latent_width is not None:
        return latent_height, latent_width
    h = getattr(config, "height", None)
    w = getattr(config, "width", None)
    if h is None or w is None:
        raise ValueError(
            "latent_height/latent_width were not provided and the mflux Config passed to "
            "call_in_loop does not expose .height/.width to auto-detect them; pass "
            "latent_height= and latent_width= explicitly."
        )
    return int(h) // downscale, int(w) // downscale


class LivePreviewCallback(InLoopCallback):  # type: ignore[misc]
    """mflux callback that writes a low-quality preview image every N iterations.

    Drops into `mflux.Flux2Klein.generate_image(callbacks=[...])`. On the
    iteration indices that match `every`, unpacks the in-flight latent, runs
    the kernel's decode (TAEF2 for FLUX.2; TAEF1 for FLUX.1 and Z-Image), and
    writes a PIL image to disk.

    Args:
        flux: optional reference to the mflux model instance the callback will be
            registered on. Typed as `object` to keep this module import-clean of
            mflux; intended to be a `Flux2Klein` instance when `auto_bn=True` and
            `variant="taef2"`. Required for auto-bn extraction (the mflux callback
            contract does not pass the flux instance at fire time).
        auto_bn: TAEF2-only. When True (default) and a `flux` instance is passed for
            `variant='taef2'`, the VAE BN running stats (and eps) are auto-extracted for
            color-correct previews. For 'taef1'/'zimage' it is a no-op (those have no BN
            step) and logs an info line so the no-op is observable. Explicit `bn_mean`/
            `bn_var` always take precedence.
        variant: 'taef1' (FLUX.1), 'taef2' (FLUX.2 Klein), or 'zimage' (Z-Image /
            Z-Image-Turbo, which reuses TAEF1's weights).
        every: emit a preview every Nth iteration. Default 5. When
            `numbered_frames=True` this is forced to 1 so the gallery
            captures every step.
        save_to: filesystem path to write previews. In single-frame mode
            (default) it's overwritten each emission. In numbered-frame
            mode it's used as a template — `step{NN}` is inserted before
            the extension and `saved_paths` lists every written file.
        numbered_frames: when True, emit one image per step into a
            gallery (`<stem>_step00.<ext>`, `<stem>_step01.<ext>`, …)
            instead of overwriting a single path. Used by the v0.2.0
            showcase to build a per-step gallery.
        latent_height: latent spatial height. Default None auto-detects it from the mflux
            Config at generation time (image_height // 16 for FLUX). Pass both latent_height
            and latent_width to override; passing exactly one raises ValueError.
        latent_width: latent spatial width; None auto-detects (see latent_height).
        bn_mean: optional BN running_mean for TAEF2 (see `unpack_flux2_latent`).
        bn_var: optional BN running_var for TAEF2.
    """

    def __init__(
        self,
        *,
        flux: object | None = None,
        auto_bn: bool = True,
        variant: str = "taef2",
        every: int = 5,
        save_to: str | Path = "preview.png",
        numbered_frames: bool = False,
        latent_height: int | None = None,
        latent_width: int | None = None,
        bn_mean: mx.array | None = None,
        bn_var: mx.array | None = None,
    ) -> None:
        """Initialise LivePreviewCallback. See class docstring for argument descriptions."""
        if (latent_height is None) != (latent_width is None):
            raise ValueError(
                "latent_height and latent_width must be set together: pass both to override "
                "the auto-detected resolution, or neither to auto-detect from the mflux "
                f"Config. Got latent_height={latent_height!r}, latent_width={latent_width!r}."
            )
        if variant == "taef1":
            self.model: Taef = TAEF1.from_pretrained(include_encoder=False)
        elif variant == "taef2":
            self.model = TAEF2.from_pretrained(include_encoder=False)
        elif variant == "zimage":
            self.model = ZImage.from_pretrained(include_encoder=False)
        else:
            raise ValueError(f"variant must be 'taef1', 'taef2', or 'zimage', got {variant!r}")
        _binding = self.model._kernel.integration
        assert _binding is not None, f"kernel {self.model._kernel.name!r} has no mflux binding"
        self._packed_downscale: int | None = _binding.packed_latent_downscale
        self.flux = flux
        self.auto_bn = auto_bn
        # Numbered-frame mode emits every step (galleries capture progression);
        # caller's `every` is honored only in single-frame mode.
        self.numbered_frames = numbered_frames
        self.every = 1 if numbered_frames else every
        self.save_to = Path(save_to)
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.saved_paths: list[Path] = []
        self.bn_eps: float = 1e-4
        # Resolve BN source. Precedence:
        #   explicit (user passed bn_mean + bn_var)
        #     > auto (auto_bn=True + variant=="taef2" + flux.vae.bn extractable)
        #     > none (identity BN, v0.1.x behavior)
        if bn_mean is not None and bn_var is not None:
            self.resolved_bn = "explicit"
        elif auto_bn and variant == "taef2" and flux is not None:
            extracted_mean, extracted_var, extracted_eps = _try_extract_bn(flux)
            if extracted_mean is not None and extracted_var is not None:
                self.bn_mean = extracted_mean
                self.bn_var = extracted_var
                if extracted_eps is not None:
                    self.bn_eps = float(extracted_eps)
                self.resolved_bn = "auto"
            else:
                logger.warning(
                    "auto_bn=True but flux instance does not expose "
                    ".vae.bn.running_mean / running_var; falling back to "
                    "identity BN (previews will be color-shifted on taef2). "
                    "Pass bn_mean= and bn_var= explicitly, or check the "
                    "mflux Flux2VAE.bn attribute path."
                )
                self.resolved_bn = "none"
        else:
            if auto_bn and flux is not None and variant != "taef2":
                logger.info(
                    "auto_bn=True is a no-op for variant=%r: BN denormalization is "
                    "TAEF2-only, so previews use identity BN (correct for %s — it has "
                    "no BN step).",
                    variant,
                    variant,
                )
            self.resolved_bn = "none"
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
        """Decode latent + save image every Nth iteration (counted by our own iter, not `t`)."""
        idx = self._iter
        self._iter += 1
        if idx % self.every != 0:
            return  # pragma: no cover

        from mlx_taef.kernels import UnpackContext

        binding = self.model._kernel.integration
        if binding is None:
            raise ValueError(f"kernel {self.model._kernel.name!r} has no mflux binding")
        if self._packed_downscale is None:
            # In-loop latent is not packed (Z-Image): unpack reads dims from the latent shape.
            # Pass the explicit dims if given, else 0 — either way the unpack ignores them.
            lh = self.latent_height if self.latent_height is not None else 0
            lw = self.latent_width if self.latent_width is not None else 0
        else:
            lh, lw = _resolve_latent_dims(
                self.latent_height, self.latent_width, config, self._packed_downscale
            )
        ctx = UnpackContext(
            latent_height=lh,
            latent_width=lw,
            bn_mean=self.bn_mean,
            bn_var=self.bn_var,
            bn_eps=self.bn_eps,
        )
        img = self.model.decode_image(binding.unpack(latents, ctx))
        target = self._resolve_target(idx)
        self._save_image(img[0], target)
        self.saved_paths.append(target)

    def _resolve_target(self, idx: int) -> Path:
        """Return the output path for emission index `idx`.

        Single-frame mode → `self.save_to` (overwritten).
        Numbered-frame mode → `<stem>_step{NN}<suffix>` next to save_to.
        """
        if not self.numbered_frames:
            return self.save_to
        return self.save_to.with_name(f"{self.save_to.stem}_step{idx:02d}{self.save_to.suffix}")

    def _save_image(self, img_nhwc_uint8: mx.array, target: Path) -> None:
        from PIL import Image

        arr = np.array(img_nhwc_uint8)
        # Lazy mkdir for numbered-frame galleries where the target dir
        # may not exist yet.
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(target)


__all__ = ["LivePreviewCallback", "unpack_flux2_latent"]
