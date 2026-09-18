"""mflux integration for mlx-taef.

Provides:
- `unpack_flux2_latent`: convert mflux's packed FLUX.2 latents to TAEF2-compatible NHWC.
- `LivePreviewCallback`: drop-in mflux callback that writes preview PNGs every N steps.
  Supports FLUX.1 (``variant='taef1'``), FLUX.2 Klein (``variant='taef2'``), Z-Image /
  Z-Image-Turbo (``variant='zimage'``, reuses TAEF1 weights), Qwen-Image / Qwen-Image-Edit
  (``variant='qwen-image'``, via taew2.1), and Krea 2 (``variant='krea2'``, shares the
  qwen-image taew2.1 weights).

Install with: `pip install "mlx-taef[mflux]"`.
"""

import importlib
import logging
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_taef.errors import MfluxNotInstalledError, UnsupportedMfluxModelError

try:
    importlib.import_module("mflux.callbacks.callback")
except ImportError as e:  # pragma: no cover
    raise MfluxNotInstalledError() from e

from mlx_taef.api import TAEF1, TAEF2, Krea2, QwenImage, Taef, ZImage

logger = logging.getLogger(__name__)

# mflux-capable preview variants: kernel name -> API class. (taesd/taesdxl have no mflux binding.)
_VARIANT_CLASSES: dict[str, type[Taef]] = {
    "taef1": TAEF1,
    "taef2": TAEF2,
    "zimage": ZImage,
    "qwen-image": QwenImage,
    "krea2": Krea2,
}


def unpack_flux2_latent(
    packed: mx.array,
    *,
    latent_height: int,
    latent_width: int,
    bn_mean: mx.array | None = None,
    bn_var: mx.array | None = None,
    bn_eps: float = 1e-4,
) -> mx.array:
    """Unpack mflux's packed `(B, N, 128)` FLUX.2 latent to the NHWC `(B, 2h, 2w, 32)` TAEF2 takes.

    Leave `bn_mean` / `bn_var` unset: TAEF2 decodes the normalized latent as mflux produces it.
    Passing the Flux2VAE batch-norm statistics applies the VAE's inverse first, which lowers
    fidelity against the full VAE decode; the parameters remain for callers that depend on it.
    Delegates to `mlx_taef.kernels.flux.unpack_flux2_latent`.
    """
    if (bn_mean is None) != (bn_var is None):
        raise ValueError("bn_mean and bn_var must be set together")
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


def _infer_variant(flux: object | None) -> str:
    """Pick the variant for `flux` from its mflux `model_config`; taef2 when there is no model.

    With no model in hand (pure-decoder use) the documented default stays `taef2`. With a
    model, the kernel registry resolves it from `model_config`; an unknown model raises
    `UnsupportedMfluxModelError` before any weights load, naming the `variant=` override.
    """
    if flux is None:
        return "taef2"
    from mlx_taef.kernels import resolve_kernel_from_model_config

    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        raise UnsupportedMfluxModelError(
            f"cannot infer the preview variant: {type(flux).__name__!r} has no usable "
            "model_config (every mflux model sets one before generating). Pass variant= "
            "explicitly."
        )
    return resolve_kernel_from_model_config(model_config).name


class LivePreviewCallback:
    """mflux callback that writes a low-quality preview image every N iterations.

    Register with ``model.callbacks.register(preview)`` before calling
    ``model.generate_image(...)``. On iteration indices matching `every`, the callback
    unpacks the in-flight latent, runs the tiny decoder, and writes an image to disk.
    Each emission creates a periodic unified-memory and I/O spike; raise `every` or omit
    the callback when memory headroom is tight.

    Args:
        flux: optional reference to the mflux model instance the callback will be
            registered on. Read at construction to infer `variant` from the model's
            `model_config` (when `variant` is not given) and, with `auto_bn=True`, to extract
            the Flux2VAE batch-norm statistics; the mflux callback contract does not pass the
            model at fire time. Typed as `object` to keep this module import-clean of mflux.
        auto_bn: TAEF2-only, default False. TAEF2 decodes the normalized latent mflux hands
            the callback, so by default no batch-norm statistics are applied. Setting it to
            True (with a `flux` instance, `variant='taef2'`) extracts the VAE BN running stats
            and eps and applies the batch-norm inverse first, as the full VAE does. That is
            measured to give lower fidelity against the full VAE decode (SSIM 0.59 vs 0.92 on
            a 512x512 FLUX.2 Klein base 4B latent) and logs a warning; it remains for callers who
            relied on it. For other variants it is a no-op and logs an info line. Explicit
            `bn_mean`/`bn_var` take precedence and carry the same warning.
        variant: which tiny decoder to run. Default None: with `flux` given, it is inferred
            from the model's `model_config` (FLUX.1 family -> 'taef1', FLUX.2 Klein ->
            'taef2', Z-Image -> 'zimage', Qwen-Image / Qwen-Image-Edit -> 'qwen-image',
            Krea 2 -> 'krea2'); a model outside those families raises
            `UnsupportedMfluxModelError` before any weights load. With no `flux`, None means
            'taef2'. An explicit value always wins: 'taef1' (FLUX.1), 'taef2' (FLUX.2 Klein),
            'zimage' (Z-Image / Z-Image-Turbo, which reuses TAEF1's weights), 'qwen-image'
            (via taew2.1), or 'krea2' (which shares the qwen-image taew2.1 weights). The
            resolved choice is readable as `callback.variant`.
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
            and latent_width to override; passing exactly one raises ValueError. Ignored (with
            a logged info line) for a variant whose in-loop latent is not packed — currently
            'zimage' and 'krea2' — since their unpack reads spatial dims from the latent's own
            shape instead.
        latent_width: latent spatial width; None auto-detects (see latent_height).
        bn_mean: optional Flux2VAE BN running_mean; opts in to the batch-norm inverse for
            TAEF2 (lower fidelity than the default, see `auto_bn`). Set together with `bn_var`.
        bn_var: optional Flux2VAE BN running_var; see `bn_mean`.
        bn_eps: epsilon for explicit BN statistics. Default 1e-4. Auto-extracted model
            statistics use the model's epsilon when available.
        on_error: runtime emission policy. ``"disable"`` logs the first failure and disables
            previews for the current generation; ``"raise"`` propagates it. Constructor
            validation always raises regardless of this setting.
    """

    def __init__(
        self,
        *,
        flux: object | None = None,
        auto_bn: bool = False,
        variant: Literal["taef1", "taef2", "zimage", "qwen-image", "krea2"] | None = None,
        every: int = 5,
        save_to: str | Path = "preview.png",
        numbered_frames: bool = False,
        latent_height: int | None = None,
        latent_width: int | None = None,
        bn_mean: mx.array | None = None,
        bn_var: mx.array | None = None,
        bn_eps: float = 1e-4,
        on_error: Literal["disable", "raise"] = "disable",
    ) -> None:
        """Initialise LivePreviewCallback. See class docstring for argument descriptions."""
        if (latent_height is None) != (latent_width is None):
            raise ValueError(
                "latent_height and latent_width must be set together: pass both to override "
                "the auto-detected resolution, or neither to auto-detect from the mflux "
                f"Config. Got latent_height={latent_height!r}, latent_width={latent_width!r}."
            )
        if every < 1:
            raise ValueError(f"every must be a positive integer (>= 1), got {every!r}.")
        if (bn_mean is None) != (bn_var is None):
            raise ValueError(
                "bn_mean and bn_var must be set together: pass both to configure explicit BN "
                "denormalization, or neither to auto-extract / use identity BN. Got "
                f"bn_mean={'set' if bn_mean is not None else None}, "
                f"bn_var={'set' if bn_var is not None else None}."
            )
        if bn_eps <= 0:
            raise ValueError(f"bn_eps must be positive, got {bn_eps!r}.")
        if on_error not in ("disable", "raise"):
            raise ValueError(f"on_error must be 'disable' or 'raise', got {on_error!r}.")
        resolved_variant: str = _infer_variant(flux) if variant is None else variant
        try:
            model_cls = _VARIANT_CLASSES[resolved_variant]
        except KeyError:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANT_CLASSES)}, got {resolved_variant!r}"
            ) from None
        self.model: Taef = model_cls.from_pretrained(include_encoder=False)
        _binding = self.model._kernel.integration
        assert _binding is not None, f"kernel {self.model._kernel.name!r} has no mflux binding"
        self._packed_downscale: int | None = _binding.packed_latent_downscale
        self.flux = flux
        self.auto_bn = auto_bn
        self._variant: str = resolved_variant
        self.on_error = on_error
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
        self.bn_eps = bn_eps
        if (
            self._packed_downscale is None
            and latent_height is not None
            and latent_width is not None
        ):
            logger.info(
                "latent_height/latent_width are ignored for variant=%r: its in-loop latent "
                "is not packed, so the unpack reads spatial dims from the latent's own shape.",
                resolved_variant,
            )
        # Resolve BN source. The default is "none": TAEF2 scores far better against the full
        # VAE on the normalized latent mflux hands the callback. Opt-in precedence:
        #   explicit (user passed bn_mean + bn_var)
        #     > auto (auto_bn=True + variant=="taef2" + flux.vae.bn extractable)
        if bn_mean is not None and bn_var is not None:
            self.resolved_bn = "explicit"
        elif auto_bn and resolved_variant == "taef2" and flux is not None:
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
                    ".vae.bn.running_mean / running_var; decoding the normalized latent "
                    "instead (the default, and the higher-fidelity path for taef2)."
                )
                self.resolved_bn = "none"
        else:
            if auto_bn and resolved_variant == "taef2":
                logger.warning(
                    "auto_bn=True for taef2 but no flux instance was provided, so no "
                    "statistics can be extracted; decoding the normalized latent instead "
                    "(the default, and the higher-fidelity path for taef2)."
                )
            elif auto_bn and flux is not None:
                logger.info(
                    "auto_bn=True is a no-op for variant=%r: BN denormalization is "
                    "TAEF2-only, so previews use identity BN (correct for %s — it has "
                    "no BN step).",
                    resolved_variant,
                    resolved_variant,
                )
            self.resolved_bn = "none"
        if self.resolved_bn != "none" and resolved_variant != "taef2":
            logger.info(
                "bn_mean/bn_var are ignored for variant=%r: only the FLUX.2 (taef2) unpack reads "
                "batch-norm statistics.",
                resolved_variant,
            )
        elif self.resolved_bn != "none":
            logger.warning(
                "Applying the Flux2VAE batch-norm inverse before TAEF2 (resolved_bn=%r) gives "
                "lower fidelity against the full VAE decode than the default normalized latent "
                "(measured SSIM 0.59 vs 0.92). Drop auto_bn=True / bn_mean / bn_var unless you "
                "depend on the old look.",
                self.resolved_bn,
            )
        self._iter = 0
        self._disabled = False

    @property
    def variant(self) -> str:
        """The tiny-decoder variant in use: the one passed, or the one inferred from `flux`."""
        return self._variant

    def call_before_loop(
        self,
        seed: object,
        prompt: object,
        latents: mx.array,
        config: object,
        canny_image: object = None,
        depth_image: object = None,
        control_images: object = None,
        **_future_hook_kwargs: object,
    ) -> None:
        """Reset per-generation state so a callback reused across multiple generate_image calls.

        Restarts its `every` cadence and numbered-frame gallery from step 0. mflux keeps
        registered callbacks on the model's persistent CallbackRegistry, so one instance fires
        across every generation; mflux dispatches this hook to any subscriber exposing
        `call_before_loop`, before each denoise loop. Single-generation behavior is unchanged.
        In numbered-frame mode a subsequent generation restarts at step00 and overwrites the
        prior gallery — use a fresh `save_to` per generation to keep both.

        The conditioning-image parameters are accepted and ignored. mflux's GenerationContext
        passes every one it knows as a keyword to every subscriber (`canny_image` and
        `depth_image` since 0.17, `control_images` since 0.19.0 for Z-Image ControlNet), so a
        callback missing one fails with TypeError before the first denoise step; the catch-all
        absorbs whichever keyword the next conditioning family adds.
        """
        self._iter = 0
        self.saved_paths = []
        self._disabled = False

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
        if self._disabled:
            return
        if idx % self.every != 0:
            return  # pragma: no cover

        if self.on_error == "raise":
            self._emit_preview(idx, latents, config)
            return
        try:
            self._emit_preview(idx, latents, config)
        except Exception as e:
            self._disabled = True
            logger.warning(
                "live preview failed at step %d: %s; disabling previews for this generation",
                idx,
                e,
            )

    def _emit_preview(self, idx: int, latents: mx.array, config: object) -> None:
        """Decode and save one preview emission."""
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
