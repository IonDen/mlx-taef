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
    return parser


def _write_sha256_sidecar(target: Path) -> Path:
    """Write a `<target>.sha256` file next to `target`.

    The content is the SHA-256 of target's bytes plus the basename, in the
    standard `<hex>  <filename>` format that `shasum -c` understands.
    """
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n")
    return sidecar


def _run_mflux_generation_and_extract_latent(variant: str, prompt: str, seed: int, height: int, width: int) -> mx.array:
    """Run one mflux generation, return the final pre-decode latent.

    Heavy MLX path. Mocked in tests via patch.object.
    """
    # Lazy-import mflux at runtime — keeps this module import-clean.
    if variant == "flux1-dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name("dev", quantize=4)
        latent = _capture_flux1_latent(flux, prompt=prompt, seed=seed, height=height, width=width)
    elif variant == "flux2-klein-base-4b":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
        latent = _capture_flux2_latent(flux, prompt=prompt, seed=seed, height=height, width=width)
    else:  # pragma: no cover
        raise ValueError(f"unsupported variant: {variant}")
    return latent


def _capture_flux1_latent(flux: object, *, prompt: str, seed: int, height: int, width: int) -> mx.array:
    """Run flux1 generation and intercept the final latent via a callback.

    Heavy MLX path; implementation lands in the bench-day commit.
    """
    raise NotImplementedError(
        "Heavy MLX path — implementation lands in the bench-day commit."
    )


def _capture_flux2_latent(flux: object, *, prompt: str, seed: int, height: int, width: int) -> mx.array:
    """FLUX.2 equivalent of _capture_flux1_latent."""
    raise NotImplementedError(
        "Heavy MLX path — implementation lands in the bench-day commit."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    latent = _run_mflux_generation_and_extract_latent(
        variant=args.variant,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )

    # Filename uses underscore separator (filesystem-safe) regardless of variant naming.
    safe_name = args.variant.replace("-", "_")
    target = args.out_dir / f"{safe_name}.safetensors"

    mx.save_safetensors(str(target), {"latent": latent})
    sidecar = _write_sha256_sidecar(target)

    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
