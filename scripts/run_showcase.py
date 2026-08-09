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
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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
    TaefError,
)
from scripts.bench_decode import _repo_relative  # noqa: E402

logger = logging.getLogger("mlx_taef.showcase")


SCHEMA_VERSION = 2


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
            "Trash with a dated suffix (generated artifacts are moved to the Trash, never deleted)."
        ),
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help=(
            "Skip LPIPS scoring in vs-VAE scenarios (SSIM only). Use when the "
            "'lpips' package (fixtures group) isn't installed, or to avoid the "
            "torchvision AlexNet weight download."
        ),
    )
    parser.add_argument("--live-worker", choices=sorted(_LIVE_SCENARIOS), help=argparse.SUPPRESS)
    parser.add_argument("--live-result", type=Path, help=argparse.SUPPRESS)
    return parser


def _move_prior_artifacts_to_trash(artifacts_dir: Path) -> Path | None:
    """Move an existing artifacts dir to ~/.Trash with a dated tag.

    Returns the new Trash path on success, None if there was nothing to
    move. Prior bench output is moved to the Trash, never deleted — re-running the
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
    lpips_result = (
        {"lpips_per_pair": [], "lpips_median": None}
        if args.no_lpips or not (taef_webps and vae_webps)
        else _compute_lpips(refs=vae_webps, cands=taef_webps)
    )

    return {
        "status": "ok",
        "fixture_integrity": integrity,
        "taef": taef_result,
        "vanilla_vae": vae_result,
        **ssim,
        **lpips_result,
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
        wired_gb = _resolve_override_wired_gb(args.cap_gb, device_wired_gb)
        if wired_gb is not None:
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
        on_error="raise",
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
    generated.image.save(final_path, "WEBP", quality=92)
    _validate_live_artifacts(callback.saved_paths, final_path, expected_count=num_steps)

    if handle is not None:
        teacache_stats = {
            "skipped_count": handle.stats.skipped_count,
            "computed_count": handle.stats.computed_count,
            "variant_id": getattr(handle, "variant_id", "unknown"),
        }

    return {
        "status": "ok",
        "scenario_dir": _repo_relative(save_dir),
        "elapsed_s": elapsed_s,
        "peak_memory_gb": peak_gb,
        "applied_cap_gb": applied_cap_gb,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "prompt": prompt,
        "height": height,
        "width": width,
        "preview_paths": [_repo_relative(p) for p in callback.saved_paths],
        "preview_count": len(callback.saved_paths),
        "final_path": _repo_relative(final_path),
        "teacache": teacache_stats,
    }


def _validate_live_artifacts(
    preview_paths: list[Path], final_path: Path, *, expected_count: int
) -> None:
    """Require a complete, non-empty preview gallery and final image."""
    if len(preview_paths) != expected_count:
        raise TaefError(f"expected {expected_count} preview frames, got {len(preview_paths)}")
    for path in [*preview_paths, final_path]:
        if not path.is_file() or path.stat().st_size == 0:
            raise TaefError(f"missing or empty live artifact: {path}")


_SCENARIO_DISPATCH = {
    "taef2_vs_vae": _run_taef2_vs_vae,
    "taef1_vs_vae": _run_taef1_vs_vae,
    "zimage_vs_vae": _run_zimage_vs_vae,
    "live_preview": _run_live_preview,
    "zimage_live_preview": _run_zimage_live_preview,
    "combined": _run_combined,
}

_LIVE_SCENARIOS = frozenset({"live_preview", "zimage_live_preview", "combined"})
_LIVE_WALL_BUDGET_S = 3300.0
_MEMORY_HEADROOM_BYTES = 4 * 1024**3


def _all_scenario_order() -> list[str]:
    """Scenario run order for `--scenario all` (a single scenario name is unaffected).

    Live scenarios run first. Each vs-VAE scenario builds LPIPS's torch+AlexNet model
    (~730 MB resident, never returned to the OS) in THIS orchestrator process; a live
    scenario's watchdog computes its memory ceiling from the device's total
    `memory_size`, so running vs-VAE first would leave the live subprocess (largest:
    zimage_live_preview) with ~730 MB less real headroom than that math assumes.
    Derived from `_SCENARIO_DISPATCH`/`_LIVE_SCENARIOS` rather than a hardcoded list so a
    newly-added scenario is placed correctly without touching this function.
    """
    live = [name for name in _SCENARIO_DISPATCH if name in _LIVE_SCENARIOS]
    vs_vae = [name for name in _SCENARIO_DISPATCH if name not in _LIVE_SCENARIOS]
    return live + vs_vae


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
        "mlx_taef_version": _detect_source_version(),
        "mlx_taef_distribution_version": _safe_version("mlx-taef") or "unknown",
        "mlx_teacache_version": _safe_version("mlx-teacache"),
        "mflux_version": _safe_version("mflux") or "unknown",
        "mlx_version": _safe_version("mlx") or "unknown",
        "python_version": platform.python_version(),
        "quantize": 4,
        "generation_dtype": "bf16",
        "taef_decode_dtype": "float32",
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


def _detect_source_version() -> str:
    """Return a git-derived version tied to the source being benchmarked."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--always"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return "unknown"


def _migrate_report_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Deterministically adapt the committed v0.6.2 report to schema v2."""
    migrated = dict(data)
    hardware = dict(migrated.get("hardware", {}))
    if "dtype" in hardware:
        hardware.setdefault("generation_dtype", hardware.pop("dtype"))
    hardware.setdefault("taef_decode_dtype", "float32")
    hardware.setdefault(
        "mlx_taef_distribution_version", hardware.get("mlx_taef_version", "unknown")
    )
    migrated["hardware"] = hardware
    scenarios: dict[str, Any] = {}
    for name, raw_scenario in migrated.get("scenarios", {}).items():
        scenario = dict(raw_scenario) if isinstance(raw_scenario, dict) else raw_scenario
        if isinstance(scenario, dict) and isinstance(scenario.get("preview_paths"), list):
            scenario.setdefault("preview_count", len(scenario["preview_paths"]))
        scenarios[name] = scenario
    migrated["scenarios"] = scenarios
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") == 1:
        return _migrate_report_v1_to_v2(data)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"unknown schema_version: got {data.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    return data


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def _live_watchdog_breach_reason(
    *,
    active_bytes: int,
    ceiling_bytes: int,
    elapsed_s: float,
    wall_budget_s: float,
) -> str | None:
    """Return the first live-worker safety limit that has been breached."""
    if active_bytes >= ceiling_bytes:
        return "memory_ceiling"
    if elapsed_s > wall_budget_s:
        return "wall_budget"
    return None


def _commit_watchdog_abort(
    result_path: Path, payload: dict[str, Any], *, stop_event: threading.Event
) -> None:
    """Durably record a watchdog abort, then kill the worker with exit code 70.

    Skips entirely when generation already signalled completion (`stop_event` set),
    so a breach observed in the same instant can't overwrite a real result. The
    exit still fires even if the abort record can't be written — dying without an
    artifact beats surviving past the safety ceiling.
    """
    if stop_event.is_set():
        return
    try:
        _write_report(result_path, payload)
    finally:
        os._exit(70)


def _resolve_override_wired_gb(cap_gb: int, device_wired_gb: int) -> int | None:
    """Clamp an operator `--cap-gb` override to the device's safe wired ceiling.

    Returns None when the device reports no Metal working-set size — in that case
    no wired limit is applied at all rather than trusting the raw override.
    """
    if not device_wired_gb:
        return None
    return min(cap_gb, device_wired_gb)


class _LiveWatchdog:
    """Cooperatively stop a live-worker watchdog thread after generation."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop_event = stop_event
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)


def _install_live_watchdog(
    result_path: Path,
    scenario: str,
    *,
    interval_s: float = 0.5,
    wall_budget_s: float = _LIVE_WALL_BUDGET_S,
) -> _LiveWatchdog:
    """Abort a live worker before it exhausts unified memory or its wall budget."""
    import mlx.core as mx

    memory_size = int(mx.device_info().get("memory_size", 0))
    if memory_size <= _MEMORY_HEADROOM_BYTES:
        raise TaefError(f"could not establish a safe memory ceiling from {memory_size} bytes")
    ceiling_bytes = memory_size - _MEMORY_HEADROOM_BYTES
    stop_event = threading.Event()
    started = time.monotonic()

    def _watch() -> None:
        while not stop_event.wait(interval_s):
            elapsed_s = time.monotonic() - started
            active_bytes = int(mx.get_active_memory())
            reason = _live_watchdog_breach_reason(
                active_bytes=active_bytes,
                ceiling_bytes=ceiling_bytes,
                elapsed_s=elapsed_s,
                wall_budget_s=wall_budget_s,
            )
            if reason is None:
                continue
            _commit_watchdog_abort(
                result_path,
                {
                    "status": "aborted",
                    "scenario": scenario,
                    "reason": reason,
                    "active_memory_bytes": active_bytes,
                    "ceiling_bytes": ceiling_bytes,
                    "elapsed_s": elapsed_s,
                    "wall_budget_s": wall_budget_s,
                },
                stop_event=stop_event,
            )

    thread = threading.Thread(target=_watch, name=f"{scenario}-watchdog", daemon=True)
    thread.start()
    return _LiveWatchdog(stop_event, thread)


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
# LPIPS (orchestrator-side, after webps are saved)
# ---------------------------------------------------------------------------


def _require_lpips() -> Any:
    """Lazily import the `lpips` package, raising a clear error if unavailable.

    `lpips` is intentionally NOT a runtime or `test`-group dependency — it drags
    torch — so it lives in the `fixtures` group only (see pyproject.toml). This
    keeps CI's `--group test` install torch-free while still letting the
    showcase report LPIPS when the operator has opted in.
    """
    try:
        import lpips
        import torch  # noqa: F401  (imported to surface an absent-torch failure here too)
    except ImportError as e:
        raise TaefError(
            "LPIPS requested but the 'lpips' package is not installed; run "
            "`uv sync --group fixtures` or pass --no-lpips"
        ) from e
    return lpips


def _build_lpips_score_fn() -> Callable[[Path, Path], float]:
    """Build the canonical LPIPS(net="alex") scorer.

    Downloads ImageNet-pretrained torchvision AlexNet weights on first use
    (network I/O) — never use `pnet_rand=True`, which skips the download but is
    no longer canonical LPIPS.
    """
    lpips = _require_lpips()
    import numpy as np
    import torch
    from PIL import Image

    loss_fn = lpips.LPIPS(net="alex")

    def _to_tensor(path: Path) -> torch.Tensor:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        img = img * 2.0 - 1.0  # LPIPS expects CHW in [-1, 1]
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def _score(ref: Path, cand: Path) -> float:
        with torch.no_grad():
            return float(loss_fn(_to_tensor(ref), _to_tensor(cand)))

    return _score


def _compute_lpips(
    refs: list[Path],
    cands: list[Path],
    *,
    score_fn: Callable[[Path, Path], float] | None = None,
) -> dict[str, Any]:
    """Compute LPIPS for each (ref, cand) pair in the cross-product.

    Mirrors `_compute_ssim`'s pairing (every ref against every cand, not a
    zip). `score_fn` is the injectable model boundary: offline tests pass a
    deterministic fake; `score_fn=None` builds the canonical, network-dependent
    scorer via `_build_lpips_score_fn`.
    """
    if score_fn is None:
        score_fn = _build_lpips_score_fn()

    per_pair: list[float] = [score_fn(ref, cand) for ref in refs for cand in cands]

    import statistics

    return {
        "lpips_per_pair": per_pair,
        "lpips_median": statistics.median(per_pair) if per_pair else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_live_scenario_subprocess(scenario: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one live scenario in a fresh process and load its durable partial result."""
    partial_dir = args.report.parent / f".{args.report.stem}-partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    result_path = partial_dir / f"{scenario}.json"
    _write_report(result_path, {"status": "running", "scenario": scenario})
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--live-worker",
        scenario,
        "--live-result",
        str(result_path),
        "--no-trash-prior",
    ]
    if args.cap_gb is not None:
        command.extend(["--cap-gb", str(args.cap_gb)])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0:
        try:
            partial = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            partial = None
        if isinstance(partial, dict) and partial.get("status") == "aborted":
            raise TaefError(
                f"live scenario worker {scenario!r} aborted: {partial.get('reason', 'unknown')} "
                f"(active={partial.get('active_memory_bytes')} bytes, "
                f"ceiling={partial.get('ceiling_bytes')} bytes, "
                f"elapsed={partial.get('elapsed_s')}s)"
            )
        raise TaefError(
            f"live scenario worker {scenario!r} failed with exit {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    if not result_path.exists():
        raise TaefError(f"live scenario worker {scenario!r} produced no result at {result_path}")
    result: dict[str, Any] = json.loads(result_path.read_text())
    return result


def _run_scenarios(
    scenarios_to_run: list[str],
    args: argparse.Namespace,
    report: dict[str, Any],
) -> int:
    """Run each scenario, recording an error entry (not aborting) if one raises.

    Write the report to disk after EACH scenario so a later failure or interruption
    never discards results already computed (subprocess-per-rep runs are minutes each).
    """
    failures = 0
    for scenario in scenarios_to_run:
        try:
            if scenario in _LIVE_SCENARIOS:
                result = _run_live_scenario_subprocess(scenario, args)
            else:
                result = _SCENARIO_DISPATCH[scenario](args)
            report["scenarios"][scenario] = result
        except Exception as e:  # noqa: BLE001, RUF100 - deliberate broad catch per spec
            failures += 1
            logger.warning("scenario %s failed: %s", scenario, e)
            report["scenarios"][scenario] = {
                "status": "error",
                "error_type": type(e).__name__,
                "reason": str(e),
            }
        _write_report(args.report, report)
    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Builds report, dispatches scenarios, writes JSON."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.live_worker is not None:
        if args.live_result is None:
            parser.error("--live-result is required with --live-worker")
        watchdog = _install_live_watchdog(args.live_result, args.live_worker)
        try:
            result = _SCENARIO_DISPATCH[args.live_worker](args)
        finally:
            watchdog.stop()
        _write_report(args.live_result, result)
        return 0

    trashed = None
    if not args.no_trash_prior:
        trashed = _move_prior_artifacts_to_trash(_ARTIFACTS_DIR)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": _build_hardware_metadata(),
        "isolation": "subprocess-per-condition",
        "prior_artifacts_moved_to": trashed.name if trashed else None,
        "scenarios": {},
    }

    scenarios_to_run = _all_scenario_order() if args.scenario == "all" else [args.scenario]

    failures = _run_scenarios(scenarios_to_run, args, report)
    print(f"Wrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
