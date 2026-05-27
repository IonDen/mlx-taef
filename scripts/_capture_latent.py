"""One-shot helper to capture a FLUX latent for use in the showcase.

Generates one fixed-seed image and saves the final latent (before VAE
decode) to `tests/fixtures/showcase_latents/<variant>.safetensors` plus
a `.sha256` sidecar. Run this once per variant when refreshing the
showcase fixtures.

Wall-clock ETAs (M1 Max, quantize=4):
- flux1-dev: ~1-2 min
- flux2-klein-base-4b: ~7-10 min

Usage:
    uv run python scripts/_capture_latent.py --variant flux1-dev
    uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mlx.core as mx

_SUPPORTED_VARIANTS = ["flux1-dev", "flux2-klein-base-4b"]
_DEFAULT_OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "showcase_latents"


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


_DEFAULT_STEPS = {"flux1-dev": 14, "flux2-klein-base-4b": 4}
_DEFAULT_GUIDANCE = {"flux1-dev": 3.5, "flux2-klein-base-4b": 1.0}


# Wired-memory guardrails. Set BEFORE any model load (see CLAUDE.md memory rules).
_WIRED_LIMIT_BYTES = 20 * 1024**3
_MEMORY_LIMIT_BYTES = 22 * 1024**3


def _install_memory_caps() -> None:
    """Pin wired + soft memory caps. Idempotent."""
    mx.set_wired_limit(_WIRED_LIMIT_BYTES)
    mx.set_memory_limit(_MEMORY_LIMIT_BYTES)


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

    raise ValueError(f"unsupported variant: {variant}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _install_memory_caps()

    arrays = _capture(
        variant=args.variant,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        guidance=args.guidance,
    )

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
