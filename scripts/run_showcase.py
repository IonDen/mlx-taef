r"""6-scenario showcase orchestrator for mlx-taef.

Scenarios:
    taef2_vs_vae        — TAEF2 decoder vs Full FLUX.2 VAE on a saved latent
    taef1_vs_vae        — TAEF1 decoder vs Full FLUX.1 VAE on a saved latent
    zimage_vs_vae       — Z-Image decoder vs Full Z-Image VAE on a saved latent
    live_preview        — gallery of N frames from one FLUX.2 generation with
                          LivePreviewCallback registered
    zimage_live_preview — gallery of N frames from one Z-Image-Turbo generation
                          with LivePreviewCallback registered
    combined            — same as live_preview plus apply_teacache wrapper

Pre-reqs (run once per release-train):
    scripts/_capture_latent.py --variant flux1-dev
    scripts/_capture_latent.py --variant flux2-klein-base-4b
    scripts/_capture_latent.py --variant z-image-turbo

Usage:
    # Reproduce the committed report (per-condition defaults: 5 TAEF reps,
    # 3 vanilla-VAE reps). Override --reps only if you knowingly want a
    # different protocol than the one tied to the headline numbers.
    uv run python scripts/run_showcase.py --scenario all \\
        --report _artifacts/showcase_report.json
"""

import argparse
import datetime
import hashlib
import json
import logging
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

# Allow running as `python scripts/run_showcase.py` (the default invocation in
# the README) — adds the repo root to sys.path so `scripts.bench_decode`
# resolves identically to the test-suite path (`from scripts.bench_decode ...`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_taef.errors import (  # noqa: E402
    FixtureLatentMissingError,
    MlxTeacacheNotInstalledError,
    SchemaVersionError,
)

logger = logging.getLogger("mlx_taef.showcase")


SCHEMA_VERSION = 1


def _import_apply_teacache() -> Any:
    """Import mflux-teacache's apply_teacache, or raise the package-rooted error."""
    try:
        from mlx_teacache import apply_teacache
    except ImportError as e:
        raise MlxTeacacheNotInstalledError() from e
    return apply_teacache


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[*_SCENARIO_DISPATCH.keys(), "all"],
        default="all",
    )
    parser.add_argument("--reps", type=int, default=None, help="Override per-condition rep counts.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("_artifacts/showcase_report.json"),
    )
    parser.add_argument(
        "--cap-gb",
        type=int,
        default=None,
        help="Override the per-condition wired memory cap (GB).",
    )
    parser.add_argument(
        "--no-trash-prior",
        action="store_true",
        help=(
            "Don't move existing _artifacts/showcase/ to ~/.Trash before "
            "starting. By default, prior bench output is preserved in "
            "Trash with a dated suffix per the CLAUDE.md 'never rm' rule."
        ),
    )
    return parser


def _move_prior_artifacts_to_trash(artifacts_dir: Path) -> Path | None:
    """Move an existing artifacts dir to ~/.Trash with a dated tag.

    Returns the new Trash path on success, None if there was nothing to
    move. Mirrors the CLAUDE.md 'never rm' guardrail — re-running the
    bench would otherwise silently overwrite multi-minute prior work.
    """
    if not artifacts_dir.exists():
        return None
    trash = Path.home() / ".Trash"
    if not trash.exists():
        # Unusual on macOS; just leave the dir in place and warn.
        logger.warning("~/.Trash not found; leaving prior artifacts in place at %s", artifacts_dir)
        return None
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    target = trash / f"mlx-taef-showcase-{stamp}"
    artifacts_dir.rename(target)
    logger.info("moved prior artifacts to %s", target)
    return target


# ---------------------------------------------------------------------------
# Scenario dispatch
# ---------------------------------------------------------------------------


_FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "showcase_latents"
_ARTIFACTS_DIR = Path(__file__).parent.parent / "_artifacts" / "showcase"


def _vs_vae_scenario(
    *,
    taef_condition: str,
    flux_variant: str,
    latent_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Shared core for taef1_vs_vae and taef2_vs_vae.

    Orchestrates bench_decode in two condition-runs (taef + vanilla_vae),
    then computes SSIM between the produced webps.
    """
    from scripts.bench_decode import _run_orchestrator

    latent = _FIXTURE_DIR / latent_name
    integrity = _check_latent_sha(latent)
    save_dir = _ARTIFACTS_DIR / taef_condition
    save_dir.mkdir(parents=True, exist_ok=True)

    taef_reps = args.reps if args.reps is not None else 5
    vae_reps = args.reps if args.reps is not None else 3

    taef_result = _run_orchestrator(
        latent_path=latent,
        condition=taef_condition,
        reps=taef_reps,
        save_dir=save_dir / "taef",
        flux_variant=flux_variant,
        cap_gb_override=args.cap_gb,
    )
    vae_result = _run_orchestrator(
        latent_path=latent,
        condition="vanilla_vae",
        reps=vae_reps,
        save_dir=save_dir / "vae",
        flux_variant=flux_variant,
        cap_gb_override=args.cap_gb,
    )

    # SSIM: cross-product of taef webps vs vae webps (the visual delta).
    taef_webps = sorted((save_dir / "taef").glob(f"{taef_condition}_rep*.webp"))
    vae_webps = sorted((save_dir / "vae").glob("vanilla_vae_rep*.webp"))
    ssim = (
        _compute_ssim(refs=vae_webps, cands=taef_webps)
        if taef_webps and vae_webps
        else {
            "ssim_per_pair": [],
            "ssim_median": 0.0,
        }
    )

    return {
        "status": "ok",
        "fixture_integrity": integrity,
        "taef": taef_result,
        "vanilla_vae": vae_result,
        **ssim,
    }


def _run_taef2_vs_vae(args: argparse.Namespace) -> dict[str, Any]:
    return _vs_vae_scenario(
        taef_condition="taef2",
        flux_variant="flux2-klein-base-4b",
        latent_name="flux2_klein_base_4b.safetensors",
        args=args,
    )


def _run_taef1_vs_vae(args: argparse.Namespace) -> dict[str, Any]:
    return _vs_vae_scenario(
        taef_condition="taef1",
        flux_variant="flux1-dev",
        latent_name="flux1_dev.safetensors",
        args=args,
    )


def _run_zimage_vs_vae(args: argparse.Namespace) -> dict[str, Any]:
    return _vs_vae_scenario(
        taef_condition="zimage",
        flux_variant="z-image-turbo",
        latent_name="z_image_turbo.safetensors",
        args=args,
    )


def _run_live_preview(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one FLUX.2 image, save a numbered preview frame at every step."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    return _live_generation(
        args=args,
        model_factory=lambda: Flux2Klein(
            quantize=4, model_config=ModelConfig.flux2_klein_base_4b()
        ),
        callback_variant="taef2",
        latent_height_divisor=16,
        latent_width_divisor=16,
        prompt="a red apple on a wooden table",
        num_steps=4,
        guidance=1.0,
        with_teacache=False,
        auto_bn=True,
        scenario_dir="live_preview",
    )


def _run_combined(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one FLUX.2 image with apply_teacache + LivePreviewCallback."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    return _live_generation(
        args=args,
        model_factory=lambda: Flux2Klein(
            quantize=4, model_config=ModelConfig.flux2_klein_base_4b()
        ),
        callback_variant="taef2",
        latent_height_divisor=16,
        latent_width_divisor=16,
        prompt="a red apple on a wooden table",
        num_steps=4,
        guidance=1.0,
        with_teacache=True,
        auto_bn=True,
        scenario_dir="combined",
    )


def _run_zimage_live_preview(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one Z-Image-Turbo image, save numbered preview frames at every step."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

    return _live_generation(
        args=args,
        model_factory=lambda: MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo()),
        callback_variant="zimage",
        latent_height_divisor=8,
        latent_width_divisor=8,
        prompt="a red apple on a wooden table",
        num_steps=4,
        guidance=0.0,
        with_teacache=False,
        auto_bn=False,
        scenario_dir="zimage_live_preview",
    )


def _live_generation(
    *,
    args: argparse.Namespace,
    model_factory: Any,
    callback_variant: str,
    latent_height_divisor: int,
    latent_width_divisor: int,
    prompt: str,
    num_steps: int,
    guidance: float,
    with_teacache: bool,
    auto_bn: bool,
    scenario_dir: str,
) -> dict[str, Any]:
    """Run one generation with a preview gallery and optional TeaCache wrap.

    Parameterized over model factory, callback variant, prompt, steps, guidance,
    and auto_bn so each scenario supplies its own pinned recipe. Set auto_bn=True
    for TAEF2 scenarios (enables BN extraction from the mflux model for
    color-correct previews); False for all other variants.
    """
    import mlx.core as mx

    save_dir = _ARTIFACTS_DIR / scenario_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Memory cap — live generation needs the full model in memory.
    # `--cap-gb` (when set) overrides the device-aware default for
    # reproducing the showcase on machines with different ceilings.
    from mlx_taef._memory_caps import compute_safe_caps_gb, install_memory_caps

    applied_cap_gb: int | None = None
    if args.cap_gb is not None:
        device_wired_gb, _ = compute_safe_caps_gb()
        wired_gb = min(args.cap_gb, device_wired_gb) if device_wired_gb else args.cap_gb
        mem_gb = min(wired_gb + 2, 22)
        mx.set_wired_limit(wired_gb * 1024**3)
        mx.set_memory_limit(mem_gb * 1024**3)
        applied_cap_gb = wired_gb
    else:
        applied_cap_gb, _ = install_memory_caps()
    mx.reset_peak_memory()

    height = 512
    width = 512
    seed = 42

    flux = model_factory()

    teacache_stats: dict[str, Any] | None = None
    handle = None
    if with_teacache:
        apply_teacache = _import_apply_teacache()
        handle = apply_teacache(flux)

    from mlx_taef.integrations.mflux import LivePreviewCallback

    callback = LivePreviewCallback(
        flux=flux if auto_bn else None,
        variant=callback_variant,
        every=1,
        numbered_frames=True,
        save_to=save_dir / f"{scenario_dir}.webp",
        latent_height=height // latent_height_divisor,
        latent_width=width // latent_width_divisor,
    )
    flux.callbacks.register(callback)

    t0 = time.perf_counter()
    generated = flux.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    elapsed_s = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1024**3

    # Save the final full-VAE image too.
    final_path = save_dir / f"{scenario_dir}_final.webp"
    try:
        generated.image.save(final_path, "WEBP", quality=92)
    except Exception as e:  # pragma: no cover
        logger.warning("could not save final image: %s", e)

    if handle is not None:
        teacache_stats = {
            "skipped_count": handle.stats.skipped_count,
            "computed_count": handle.stats.computed_count,
            "variant_id": getattr(handle, "variant_id", "unknown"),
        }

    return {
        "status": "ok",
        "scenario_dir": str(save_dir),
        "elapsed_s": elapsed_s,
        "peak_memory_gb": peak_gb,
        "applied_cap_gb": applied_cap_gb,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "prompt": prompt,
        "height": height,
        "width": width,
        "preview_paths": [str(p) for p in callback.saved_paths],
        "final_path": str(final_path),
        "teacache": teacache_stats,
    }


_SCENARIO_DISPATCH = {
    "taef2_vs_vae": _run_taef2_vs_vae,
    "taef1_vs_vae": _run_taef1_vs_vae,
    "zimage_vs_vae": _run_zimage_vs_vae,
    "live_preview": _run_live_preview,
    "zimage_live_preview": _run_zimage_live_preview,
    "combined": _run_combined,
}


# ---------------------------------------------------------------------------
# JSON I/O (testable in isolation)
# ---------------------------------------------------------------------------


def _build_hardware_metadata() -> dict[str, Any]:
    """Collect hardware + version metadata for the report header.

    On macOS we query `sysctl` for the chip model and total memory so
    the report records "Apple M1 Max" / 34 GB instead of the useless
    `platform.processor() == "arm"`. Also records the git SHA so the
    benchmark is tied to an exact commit.
    """

    def _safe_version(pkg: str) -> str | None:
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "chip": _detect_chip_model(),
        "ram_gb": _detect_ram_gb(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "git_sha": _detect_git_sha(),
        "mlx_taef_version": _safe_version("mlx-taef") or "0.0.0+dev",
        "mlx_teacache_version": _safe_version("mlx-teacache"),
        "mflux_version": _safe_version("mflux") or "unknown",
        "mlx_version": _safe_version("mlx") or "unknown",
        "python_version": platform.python_version(),
        "quantize": 4,
        "dtype": "bf16",
    }


def _detect_chip_model() -> str:
    """Return e.g. 'Apple M1 Max' on macOS, fallback to platform.processor()."""
    import subprocess as _subp

    if platform.system() != "Darwin":
        return platform.processor() or "unknown"
    try:
        result = _subp.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, _subp.SubprocessError):  # pragma: no cover
        pass
    return platform.processor() or "unknown"


def _detect_ram_gb() -> int:
    """Return total system RAM in GB. macOS via sysctl, fallback to 0."""
    import subprocess as _subp

    if platform.system() != "Darwin":
        return 0
    try:
        result = _subp.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return round(int(result.stdout.strip()) / 1024**3)
    except (OSError, ValueError, _subp.SubprocessError):  # pragma: no cover
        pass
    return 0


def _detect_git_sha() -> str | None:
    """Return the current git commit SHA or None if not in a repo."""
    import subprocess as _subp

    try:
        result = _subp.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, _subp.SubprocessError):  # pragma: no cover
        pass
    return None


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"unknown schema_version: got {data.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    return data


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Latent fixture handling
# ---------------------------------------------------------------------------


def _check_latent_sha(latent: Path) -> dict[str, str]:
    """Verify the .sha256 sidecar matches; raise if missing latent.

    Returns a status dict the orchestrator records under
    `report["fixture_integrity"][<latent_stem>]`. Possible values:
    - {"status": "ok"} — hash matches
    - {"status": "no_sidecar"} — no .sha256 file present, skipped check
    - {"status": "mismatch", "expected": ..., "actual": ...} — drift
    """
    if not latent.exists():
        raise FixtureLatentMissingError(
            f"showcase latent missing at {latent}; run "
            f"`scripts/_capture_latent.py --variant <name>` first."
        )
    sidecar = latent.with_suffix(latent.suffix + ".sha256")
    if not sidecar.exists():
        logger.warning("no .sha256 sidecar at %s; skipping integrity check", sidecar)
        return {"status": "no_sidecar"}
    expected = sidecar.read_text().split()[0]
    actual = hashlib.sha256(latent.read_bytes()).hexdigest()
    if expected != actual:
        logger.warning(
            "sha mismatch for %s: expected %s, got %s. Latent may have been regenerated.",
            latent.name,
            expected[:12],
            actual[:12],
        )
        return {"status": "mismatch", "expected": expected, "actual": actual}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# SSIM (orchestrator-side, after webps are saved)
# ---------------------------------------------------------------------------


def _compute_ssim(refs: list[Path], cands: list[Path]) -> dict[str, Any]:
    """Compute SSIM for each (ref, cand) pair in the cross-product."""
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    per_pair: list[float] = []
    for ref in refs:
        ref_img = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32) / 255.0
        for cand in cands:
            cand_img = np.asarray(Image.open(cand).convert("RGB"), dtype=np.float32) / 255.0
            score = float(
                structural_similarity(
                    ref_img,
                    cand_img,
                    channel_axis=-1,
                    data_range=1.0,
                )
            )
            per_pair.append(score)
    import statistics

    return {
        "ssim_per_pair": per_pair,
        "ssim_median": statistics.median(per_pair) if per_pair else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Builds report, dispatches scenarios, writes JSON."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    trashed = None
    if not args.no_trash_prior:
        trashed = _move_prior_artifacts_to_trash(_ARTIFACTS_DIR)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "hardware": _build_hardware_metadata(),
        "isolation": "subprocess-per-rep",
        "prior_artifacts_moved_to": str(trashed) if trashed else None,
        "scenarios": {},
    }

    scenarios_to_run = (
        list(_SCENARIO_DISPATCH.keys()) if args.scenario == "all" else [args.scenario]
    )

    for scenario in scenarios_to_run:
        try:
            report["scenarios"][scenario] = _SCENARIO_DISPATCH[scenario](args)
        except NotImplementedError as e:
            logger.warning("scenario %s skipped: %s", scenario, e)
            report["scenarios"][scenario] = {"status": "not_implemented", "reason": str(e)}

    _write_report(args.report, report)
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
