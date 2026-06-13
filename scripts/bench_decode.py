"""Subprocess-per-rep decoder bench worker + orchestrator.

Reused by `scripts/run_showcase.py` for the `taef*_vs_vae` scenarios.
Standalone CLI for regression-tracking.

Sentinel contract (deliberate tightening of mlx-teacache template):
- `::BENCH_RESULT::<json>` is line-start, exactly one per worker,
  JSON one-liner.
- Multiple sentinels in one worker's stdout raise TaefError.
- Missing sentinel raises TaefError.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as `python scripts/bench_decode.py` (worker subprocesses spawn
# us this way). Adds the repo root to sys.path so `scripts._caps` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_taef.errors import TaefError  # noqa: E402  (after sys.path tweak)
from mlx_taef.variants import get_memory_cap_hint  # noqa: E402
from scripts._caps import FULL_VAE_CAP_GB  # noqa: E402

SENTINEL_PREFIX = "::BENCH_RESULT::"

# Per-condition subprocess timeouts (seconds). Full-VAE workers cold-load
# a multi-GB flux pipeline before they decode, plus possibly download
# from HF on first run; 1200s leaves margin for those. Taef workers load
# a ~4 MB decoder so 600s is generous.
_SUBPROCESS_TIMEOUT_S = {
    "taef1": 600,
    "taef2": 600,
    "zimage": 600,
    "vanilla_vae": 1200,
}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True, help="Input safetensors latent.")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["taef1", "taef2", "zimage", "vanilla_vae"],
        help="Which decoder to run.",
    )
    parser.add_argument(
        "--flux-variant",
        default="flux2-klein-base-4b",
        choices=list(FULL_VAE_CAP_GB.keys()),
        help="Which full Flux variant the vanilla_vae condition uses.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Reps for the orchestrator (default 5 for taef, 3 for full VAE).",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("_artifacts/showcase"))

    # Worker-mode flags (hidden from typical user; orchestrator passes these to itself).
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rep", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--save-to", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--applied-cap-gb", type=int, help=argparse.SUPPRESS)

    return parser


def _resolve_cap_gb(*, condition: str, flux_variant: str = "flux2-klein-base-4b") -> int | None:
    """Per-condition cap policy. See spec Section 3 'Per-condition cap policy'."""
    if condition in ("taef1", "taef2"):
        return get_memory_cap_hint(condition)
    if condition == "zimage":
        # Route via the registry — the legacy shim raises on "zimage".
        from mlx_taef.kernels import KERNELS

        return KERNELS["zimage"].memory_cap_hint_gb
    if condition == "vanilla_vae":
        return FULL_VAE_CAP_GB[flux_variant]
    raise TaefError(f"unknown condition: {condition!r}")


def _emit_sentinel(result: dict[str, Any]) -> str:
    """Format the sentinel as a single line. Caller prints it."""
    return SENTINEL_PREFIX + json.dumps(result)


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    """Extract the single sentinel from worker stdout.

    Contract: line-start, exactly one per worker, JSON one-liner.
    """
    sentinel_lines = [line for line in stdout.splitlines() if line.startswith(SENTINEL_PREFIX)]
    if len(sentinel_lines) == 0:
        raise TaefError(f"no sentinel found in worker stdout (stdout: {stdout[:500]!r})")
    if len(sentinel_lines) > 1:
        raise TaefError(
            f"multiple sentinels in worker stdout (got {len(sentinel_lines)}, expected 1)"
        )
    payload_str = sentinel_lines[0][len(SENTINEL_PREFIX) :]
    try:
        payload: dict[str, Any] = json.loads(payload_str)
    except json.JSONDecodeError as e:
        # Re-raise as TaefError so the orchestrator's TaefError funnel
        # catches it and marks the rep failed instead of aborting the
        # entire showcase run.
        raise TaefError(
            f"malformed sentinel JSON (probable partial flush): {e}; "
            f"payload was {payload_str[:200]!r}"
        ) from e
    return payload


def _run_one_rep(
    *,
    latent_path: Path,
    condition: str,
    flux_variant: str,
    rep: int,
    save_to: Path,
    cap_gb: int | None,
) -> dict[str, Any]:
    """Spawn the worker subprocess for one rep; return its parsed sentinel."""
    cmd = [
        sys.executable,
        __file__,
        "--worker-mode",
        "--latent",
        str(latent_path),
        "--condition",
        condition,
        "--flux-variant",
        flux_variant,
        "--rep",
        str(rep),
        "--save-to",
        str(save_to),
    ]
    if cap_gb is not None:
        cmd.extend(["--applied-cap-gb", str(cap_gb)])

    timeout_s = _SUBPROCESS_TIMEOUT_S.get(condition, 600)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {
            "condition": condition,
            "rep": rep,
            "status": "failed",
            "error": f"timeout after {timeout_s}s",
        }

    if proc.returncode != 0:
        # Cap rejected at startup, OOM, jetsam, etc. Parse stderr for hints.
        return {
            "condition": condition,
            "rep": rep,
            "status": "failed",
            "error": f"exit {proc.returncode}: {proc.stderr[:300]}",
        }
    try:
        return _parse_worker_stdout(proc.stdout)
    except TaefError as e:
        return {
            "condition": condition,
            "rep": rep,
            "status": "failed",
            "error": str(e),
        }


def _run_orchestrator(
    *,
    latent_path: Path,
    condition: str,
    reps: int,
    save_dir: Path,
    flux_variant: str = "flux2-klein-base-4b",
    cap_gb_override: int | None = None,
) -> dict[str, Any]:
    """Run all reps for one condition; aggregate.

    `cap_gb_override` (when set) replaces the per-condition default cap
    from `_resolve_cap_gb`. Used by `run_showcase.py --cap-gb` to let
    operators try a different cap during reproduction.
    """
    if cap_gb_override is not None:
        cap_gb: int | None = cap_gb_override
    else:
        cap_gb = _resolve_cap_gb(condition=condition, flux_variant=flux_variant)
    save_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for rep in range(reps):
        save_to = save_dir / f"{condition}_rep{rep}.webp"
        result = _run_one_rep(
            latent_path=latent_path,
            condition=condition,
            flux_variant=flux_variant,
            rep=rep,
            save_to=save_to,
            cap_gb=cap_gb,
        )
        if result.get("status") == "failed":
            failures.append(result)
        else:
            successes.append(result)

    if not successes:
        raise TaefError(f"all reps failed for condition={condition}: {failures}")

    per_rep_seconds = [r["elapsed_s"] for r in successes]
    per_rep_peak = [r.get("peak_memory_gb", 0.0) for r in successes]
    installed_caps = sorted(
        {r.get("installed_cap_gb") for r in successes if r.get("installed_cap_gb") is not None}
    )
    return {
        "condition": condition,
        "applied_cap_gb": cap_gb,
        "installed_cap_gb": installed_caps[0] if len(installed_caps) == 1 else installed_caps,
        "reps": len(successes),
        "per_rep_seconds": per_rep_seconds,
        "median_seconds": statistics.median(per_rep_seconds),
        "min_seconds": min(per_rep_seconds),
        "max_seconds": max(per_rep_seconds),
        "per_rep_peak_memory_gb": per_rep_peak,
        "median_peak_memory_gb": statistics.median(per_rep_peak),
        "image_path": str(successes[-1].get("image_path", "")),
        "per_rep_failures": failures,
    }


def _install_memory_caps(applied_cap_gb: int | None) -> int:
    """Pin wired + soft memory caps before any model load.

    `applied_cap_gb` is the per-condition cap from `_resolve_cap_gb`
    (taef1=1, taef2=2, vanilla_vae=6 or 12). When provided it overrides
    the device-aware default. When None, falls back to the hardware
    ceiling clamp from `_memory_caps.install_memory_caps`.

    Returns the wired-cap (in GB) actually installed — may be lower
    than `applied_cap_gb` if the device's max_recommended_working_set
    forced a clamp. 0 means no cap was installed (non-Metal env).
    """
    import mlx.core as mx

    from mlx_taef._memory_caps import compute_safe_caps_gb, install_memory_caps

    if applied_cap_gb is None:
        installed_wired_gb, _ = install_memory_caps()
        return installed_wired_gb

    # Clamp the requested condition cap to fit the device too — same
    # reason: a 12 GB vae cap would raise on an 8 GB CI runner.
    device_wired_gb, _ = compute_safe_caps_gb()
    if device_wired_gb == 0:
        return 0  # non-Metal env
    wired_gb = min(applied_cap_gb, device_wired_gb)
    mem_gb = min(wired_gb + 2, 22)
    mx.set_wired_limit(wired_gb * 1024**3)
    mx.set_memory_limit(mem_gb * 1024**3)
    return wired_gb


def _decode_taef1(latent: Any, height: int, width: int) -> Any:
    """Packed (1, lH*lW, 64) → NHWC uint8 (1, H, W, 3) via TAEF1."""
    import mlx.core as mx
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

    from mlx_taef.api import TAEF1

    taef = TAEF1.from_pretrained(include_encoder=False)
    unpacked_nchw = FluxLatentCreator.unpack_latents(latent, height=height, width=width)
    unpacked_nhwc = mx.transpose(unpacked_nchw, (0, 2, 3, 1))
    return taef.decode_image(unpacked_nhwc)


def _decode_taef2(latent: Any, height: int, width: int, bn_mean: Any, bn_var: Any) -> Any:
    """Packed (1, lH*lW, 128) → NHWC uint8 (1, H, W, 3) via TAEF2 with BN denorm."""
    from mlx_taef.api import TAEF2
    from mlx_taef.integrations.mflux import unpack_flux2_latent

    taef = TAEF2.from_pretrained(include_encoder=False)
    latent_h = height // 16
    latent_w = width // 16
    unpacked = unpack_flux2_latent(
        latent,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=bn_mean,
        bn_var=bn_var,
    )
    return taef.decode_image(unpacked)


def _decode_zimage(latent: Any, height: int, width: int) -> Any:
    """Packed Z-Image in-loop latent (16, 1, h, w) → NHWC uint8 (1, H, W, 3) via ZImage TAEF.

    Returns `decode_image()` directly — it already produces uint8 NHWC.
    Do NOT pass through `_decoded_to_uint8_nhwc` (that helper is for full-VAE NCHW [-1,1]
    tensors and would transpose/renormalize uint8 into garbage).
    """
    from mlx_taef import ZImage
    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.zimage import unpack_zimage_latent

    taef = ZImage.from_pretrained(include_encoder=False)
    ctx = UnpackContext(latent_height=height // 8, latent_width=width // 8)
    return taef.decode_image(unpack_zimage_latent(latent, ctx))  # already uint8 NHWC (1,H,W,3)


def _decode_full_vae_zimage(latent: Any, height: int, width: int) -> Any:
    """Packed Z-Image latent → full mflux Z-Image VAE decode → NHWC uint8 image."""
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
    from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

    model = MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo())
    unpacked = ZImageLatentCreator.unpack_latents(latent, height, width)
    decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=None)
    mx.eval(decoded)
    return _decoded_to_uint8_nhwc(decoded)


def _decode_full_vae_flux1(latent: Any, height: int, width: int) -> Any:
    """Packed → full FLUX.1 VAE decode → NHWC uint8 image."""
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    unpacked = FluxLatentCreator.unpack_latents(latent, height=height, width=width)
    decoded = flux.vae.decode(unpacked)
    return _decoded_to_uint8_nhwc(decoded)


def _decode_full_vae_flux2(latent: Any, height: int, width: int) -> Any:
    """Packed → full FLUX.2 Klein VAE decode → NHWC uint8 image."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    latent_h = height // 16
    latent_w = width // 16
    packed_nchw = latent.reshape(latent.shape[0], latent_h, latent_w, latent.shape[-1]).transpose(
        0, 3, 1, 2
    )
    decoded = flux.vae.decode_packed_latents(packed_nchw)
    return _decoded_to_uint8_nhwc(decoded)


def _decoded_to_uint8_nhwc(decoded: Any) -> Any:
    """Mirror mflux ImageUtil._denormalize + transpose + 255x scale + cast."""
    import mlx.core as mx

    # decoded may be (B, 3, 1, H, W) for flux1 or (B, 3, H, W) for flux2.
    if decoded.ndim == 5:
        decoded = mx.squeeze(decoded, axis=2)
    normalized = mx.clip(decoded / 2.0 + 0.5, 0.0, 1.0)
    nhwc = mx.transpose(normalized, (0, 2, 3, 1))
    return (nhwc * 255.0).astype(mx.uint8)


def _save_webp(image_uint8_nhwc: Any, target: Path) -> None:
    """Write a single-image batch as a webp."""
    import numpy as np
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(image_uint8_nhwc[0])
    Image.fromarray(arr).save(target, "WEBP", quality=92)


def _worker_main(args: argparse.Namespace) -> int:
    """Run one (condition, rep) inside this subprocess. Emit sentinel."""
    installed_cap_gb = _install_memory_caps(args.applied_cap_gb)

    import mlx.core as mx

    arrays = mx.load(str(args.latent))
    latent = arrays["latent"]
    height = int(arrays["height"].item())
    width = int(arrays["width"].item())
    bn_mean = arrays.get("bn_mean")
    bn_var = arrays.get("bn_var")

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if args.condition == "taef1":
        image = _decode_taef1(latent, height, width)
    elif args.condition == "taef2":
        if bn_mean is None or bn_var is None:
            raise TaefError("taef2 condition requires bn_mean+bn_var in the latent safetensors")
        image = _decode_taef2(latent, height, width, bn_mean, bn_var)
    elif args.condition == "zimage":
        image = _decode_zimage(latent, height, width)
    elif args.condition == "vanilla_vae":
        if args.flux_variant == "flux1-dev":
            image = _decode_full_vae_flux1(latent, height, width)
        elif args.flux_variant == "flux2-klein-base-4b":
            image = _decode_full_vae_flux2(latent, height, width)
        elif args.flux_variant == "z-image-turbo":
            image = _decode_full_vae_zimage(latent, height, width)
        else:  # pragma: no cover
            raise TaefError(f"unknown flux_variant: {args.flux_variant!r}")
    else:  # pragma: no cover
        raise TaefError(f"unknown condition: {args.condition!r}")
    mx.eval(image)
    elapsed_s = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1024**3

    _save_webp(image, args.save_to)

    print(
        _emit_sentinel(
            {
                "condition": args.condition,
                "rep": args.rep,
                "status": "ok",
                "elapsed_s": elapsed_s,
                "peak_memory_gb": peak_gb,
                "image_path": str(args.save_to),
                "requested_cap_gb": args.applied_cap_gb,
                "installed_cap_gb": installed_cap_gb,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Dispatches to worker or orchestrator based on --worker-mode."""
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.worker_mode:
        return _worker_main(args)
    result = _run_orchestrator(
        latent_path=args.latent,
        condition=args.condition,
        reps=args.reps,
        save_dir=args.save_dir,
        flux_variant=args.flux_variant,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
