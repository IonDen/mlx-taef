"""One-shot helper to capture a FLUX latent for use in the showcase.

Generates one fixed-seed image and saves the final latent (before VAE
decode) to `tests/fixtures/showcase_latents/<variant>.safetensors` plus
a `.sha256` sidecar. Run this once per variant when refreshing the
showcase fixtures.

Wall-clock ETAs (M1 Max, quantize=4):
- flux1-dev: ~1-2 min
- flux2-klein-base-4b: ~7-10 min
- krea-2-turbo: ~25-45 min cold cache / ~5-10 min warm (first run downloads ~36 GB of
  Krea-2-Turbo weights; observed ~25 MB/s puts the download alone at ~25 min)

A watchdog thread aborts the run (writing `<out-dir>/<variant>.abort.json` and exiting
nonzero) if active memory nears the device ceiling or the wall budget is exceeded —
same discipline as `scripts/run_showcase.py`'s `_install_live_watchdog`, sized down for a
single-shot capture instead of a multi-scenario orchestrator.

Usage:
    uv run python scripts/_capture_latent.py --variant flux1-dev
    uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b
    uv run python scripts/_capture_latent.py --variant krea-2-turbo
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx

_SUPPORTED_VARIANTS = ["flux1-dev", "flux2-klein-base-4b", "z-image-turbo", "krea-2-turbo"]
_DEFAULT_OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "showcase_latents"

# Wall budget is a backstop, not an ETA estimate: a cold-cache krea-2-turbo run downloads
# ~36 GB at an observed ~25 MB/s (~25 min) before generation even starts, plus model load +
# an 8-step generation. 3600s (60 min) leaves real margin above that ~25-45 min cold-cache
# range (a prior 1200s budget aborted a real run mid-download with memory nowhere near the
# ceiling — see krea_2_turbo.abort.json, active_memory_bytes: 128). Memory headroom mirrors
# run_showcase.py's _MEMORY_HEADROOM_BYTES (abort at memory_size - 4 GiB).
_WALL_BUDGET_S = 3600.0
_MEMORY_HEADROOM_BYTES = 4 * 1024**3


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=_SUPPORTED_VARIANTS,
        help="Flux variant to capture a latent from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory to write the safetensors + sidecar into.",
    )
    parser.add_argument(
        "--prompt",
        default="a red apple on a wooden table",
        help="Fixed prompt for reproducibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override inference steps. Defaults: flux1-dev=14, flux2-klein-base-4b=4.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Override CFG guidance. Defaults: flux1-dev=3.5, flux2-klein-base-4b=1.0.",
    )
    return parser


_DEFAULT_STEPS = {"flux1-dev": 14, "flux2-klein-base-4b": 4, "z-image-turbo": 4, "krea-2-turbo": 8}
_DEFAULT_GUIDANCE = {
    "flux1-dev": 3.5,
    "flux2-klein-base-4b": 1.0,
    # Z-Image-Turbo is distilled (supports_guidance=False); mflux force-overrides guidance to
    # 0.0 internally (z_image.py:60-62), so 0.0 is the effective value regardless of what is
    # passed.  Naming any other value would make the recipe lie about the run mflux performed.
    "z-image-turbo": 0.0,
    # Krea-2-Turbo's own CLI (mflux/models/krea2/cli/krea2_generate.py) defaults to
    # DEFAULT_STEPS=8 / DEFAULT_GUIDANCE=1.0 (er_sde scheduler) — mirrored here verbatim.
    "krea-2-turbo": 1.0,
}


def _abort_artifact_path(variant: str, out_dir: Path) -> Path:
    """Where a capture-watchdog abort artifact for `variant` lives, if one exists."""
    return out_dir / f"{variant.replace('-', '_')}.abort.json"


def _watchdog_breach_reason(
    *, active_bytes: int, ceiling_bytes: int, elapsed_s: float, wall_budget_s: float
) -> str | None:
    """Return the first capture-run safety limit that has been breached."""
    if active_bytes >= ceiling_bytes:
        return "memory_ceiling"
    if elapsed_s > wall_budget_s:
        return "wall_budget"
    return None


class _CaptureWatchdog:
    """Cooperatively stop a capture-run watchdog thread once generation finishes."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop_event = stop_event
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)


def _install_capture_watchdog(
    variant: str,
    out_dir: Path,
    *,
    interval_s: float = 0.5,
    wall_budget_s: float = _WALL_BUDGET_S,
) -> _CaptureWatchdog:
    """Abort a heavy capture run before it exhausts unified memory or its wall budget.

    Modeled on scripts/run_showcase.py's `_install_live_watchdog`: a daemon thread polls
    `mx.get_active_memory()` and elapsed wall time, and on breach writes an honest abort
    artifact (`<out-dir>/<variant>.abort.json`) before killing the process with a nonzero
    exit — no partial/misleading latent fixture is ever written.
    """
    memory_size = int(mx.device_info().get("memory_size", 0))
    if memory_size <= _MEMORY_HEADROOM_BYTES:
        raise RuntimeError(f"could not establish a safe memory ceiling from {memory_size} bytes")
    ceiling_bytes = memory_size - _MEMORY_HEADROOM_BYTES
    stop_event = threading.Event()
    started = time.monotonic()
    abort_path = _abort_artifact_path(variant, out_dir)

    def _watch() -> None:
        while not stop_event.wait(interval_s):
            elapsed_s = time.monotonic() - started
            active_bytes = int(mx.get_active_memory())
            reason = _watchdog_breach_reason(
                active_bytes=active_bytes,
                ceiling_bytes=ceiling_bytes,
                elapsed_s=elapsed_s,
                wall_budget_s=wall_budget_s,
            )
            if reason is None:
                continue
            if stop_event.is_set():
                # Generation finished in the same instant the breach was observed; don't
                # overwrite a real result with an abort artifact.
                return
            abort_path.write_text(
                json.dumps(
                    {
                        "status": "aborted",
                        "variant": variant,
                        "reason": reason,
                        "active_memory_bytes": active_bytes,
                        "ceiling_bytes": ceiling_bytes,
                        "elapsed_s": elapsed_s,
                        "wall_budget_s": wall_budget_s,
                    },
                    indent=2,
                )
            )
            os._exit(70)

    thread = threading.Thread(target=_watch, name=f"{variant}-capture-watchdog", daemon=True)
    thread.start()
    return _CaptureWatchdog(stop_event, thread)


def _install_memory_caps() -> None:
    """Pin hardware-aware wired + soft memory caps. Idempotent.

    Delegates to mlx_taef._memory_caps.install_memory_caps which clamps
    the desired (20 GB / 22 GB) targets to whatever fits the actual
    device — preserves user-machine behavior, doesn't crash CI runners.
    """
    from mlx_taef._memory_caps import install_memory_caps

    install_memory_caps()


def _write_sha256_sidecar(target: Path) -> Path:
    """Write a `<target>.sha256` file next to `target`.

    The content is the SHA-256 of target's bytes plus the basename, in the
    standard `<hex>  <filename>` format that `shasum -c` understands.
    """
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n")
    return sidecar


class _LatentCaptureCallback:
    """AfterLoopCallback that stores the final latent on self for retrieval.

    mflux's CallbackRegistry duck-types: any object with `call_after_loop` is
    registered as an AfterLoopCallback. No base class needed.
    """

    def __init__(self) -> None:
        self.latents: mx.array | None = None

    def call_after_loop(
        self,
        *,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: object,
    ) -> None:
        # Force evaluation so the captured array isn't a lazy graph reference.
        mx.eval(latents)
        self.latents = latents


def _capture_flux1_latent(
    flux: object,
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
) -> mx.array:
    """Run flux1 generation and intercept the final packed latent via callback."""
    capture = _LatentCaptureCallback()
    flux.callbacks.register(capture)  # type: ignore[attr-defined]
    flux.generate_image(  # type: ignore[attr-defined]
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    if capture.latents is None:
        raise RuntimeError("AfterLoopCallback did not fire — mflux contract changed?")
    return capture.latents


def _capture_flux2_latent(
    flux: object,
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
) -> mx.array:
    """FLUX.2 equivalent of _capture_flux1_latent."""
    capture = _LatentCaptureCallback()
    flux.callbacks.register(capture)  # type: ignore[attr-defined]
    flux.generate_image(  # type: ignore[attr-defined]
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    if capture.latents is None:
        raise RuntimeError("AfterLoopCallback did not fire — mflux contract changed?")
    return capture.latents


def _capture_zimage_latent(
    flux: object,
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
) -> mx.array:
    """Z-Image equivalent of _capture_flux1_latent.

    mflux's Z-Image in-loop latent is (16, 1, h, w).  The AfterLoopCallback
    receives the same packed shape, which is what the SSIM gate and showcase
    downstream expect.
    """
    capture = _LatentCaptureCallback()
    flux.callbacks.register(capture)  # type: ignore[attr-defined]
    flux.generate_image(  # type: ignore[attr-defined]
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    if capture.latents is None:
        raise RuntimeError("AfterLoopCallback did not fire — mflux contract changed?")
    return capture.latents


def _capture_krea2_latent(
    flux: object,
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
) -> mx.array:
    """Krea 2 equivalent of _capture_flux1_latent.

    mflux's Krea 2 in-loop latent is (1, 16, h//8, w//8) NCHW — Krea2LatentCreator's
    create_noise/pack_latents/unpack_latents are all identity, so the AfterLoopCallback
    receives the same raw shape the denoise loop produces (see kernels/krea2.py docstring).
    """
    capture = _LatentCaptureCallback()
    flux.callbacks.register(capture)  # type: ignore[attr-defined]
    flux.generate_image(  # type: ignore[attr-defined]
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    if capture.latents is None:
        raise RuntimeError("AfterLoopCallback did not fire — mflux contract changed?")
    return capture.latents


def _capture(
    variant: str,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int | None,
    guidance: float | None,
) -> dict[str, mx.array]:
    """Run generation and return arrays to be saved (latent + optional BN stats)."""
    steps = num_steps if num_steps is not None else _DEFAULT_STEPS[variant]
    cfg = guidance if guidance is not None else _DEFAULT_GUIDANCE[variant]

    if variant == "flux1-dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name("dev", quantize=4)
        latent = _capture_flux1_latent(
            flux,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_steps=steps,
            guidance=cfg,
        )
        return {
            "latent": latent,
            "height": mx.array([height], dtype=mx.int32),
            "width": mx.array([width], dtype=mx.int32),
        }

    if variant == "flux2-klein-base-4b":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
        latent = _capture_flux2_latent(
            flux,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_steps=steps,
            guidance=cfg,
        )
        # Also persist the VAE BN stats so downstream TAEF2 decoders don't
        # need to re-load Flux2Klein just to read them. See unpack_flux2_latent.
        bn_mean = mx.array(flux.vae.bn.running_mean)  # type: ignore[attr-defined]
        bn_var = mx.array(flux.vae.bn.running_var)  # type: ignore[attr-defined]
        mx.eval(bn_mean, bn_var)
        return {
            "latent": latent,
            "bn_mean": bn_mean,
            "bn_var": bn_var,
            "height": mx.array([height], dtype=mx.int32),
            "width": mx.array([width], dtype=mx.int32),
        }

    if variant == "z-image-turbo":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

        flux = MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo())
        latent = _capture_zimage_latent(
            flux,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_steps=steps,
            guidance=cfg,
        )
        assert latent.shape == (16, 1, height // 8, width // 8), latent.shape
        return {
            "latent": latent,
            "height": mx.array([height], dtype=mx.int32),
            "width": mx.array([width], dtype=mx.int32),
        }

    if variant == "krea-2-turbo":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2

        flux = Krea2(quantize=4, model_config=ModelConfig.krea2())
        latent = _capture_krea2_latent(
            flux,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_steps=steps,
            guidance=cfg,
        )
        assert latent.shape == (1, 16, height // 8, width // 8), latent.shape
        return {
            "latent": latent,
            "height": mx.array([height], dtype=mx.int32),
            "width": mx.array([width], dtype=mx.int32),
        }

    raise ValueError(f"unsupported variant: {variant}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Drop any stale abort artifact from a prior aborted run before installing the watchdog,
    # so a later successful capture never leaves a misleading abort record lingering next to it.
    _abort_artifact_path(args.variant, args.out_dir).unlink(missing_ok=True)
    _install_memory_caps()

    watchdog = _install_capture_watchdog(args.variant, args.out_dir)
    try:
        arrays = _capture(
            variant=args.variant,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance=args.guidance,
        )
    finally:
        watchdog.stop()

    # Filename uses underscore separator (filesystem-safe) regardless of variant naming.
    safe_name = args.variant.replace("-", "_")
    target = args.out_dir / f"{safe_name}.safetensors"

    mx.save_safetensors(str(target), arrays)
    sidecar = _write_sha256_sidecar(target)

    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
