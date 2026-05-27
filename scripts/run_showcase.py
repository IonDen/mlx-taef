r"""4-scenario showcase orchestrator for mlx-taef v0.2.0.

Scenarios:
    taef2_vs_vae   — TAEF2 decoder vs Full FLUX.2 VAE on a saved latent
    taef1_vs_vae   — TAEF1 decoder vs Full FLUX.1 VAE on a saved latent
    live_preview   — gallery of N frames from one FLUX.2 generation with
                     LivePreviewCallback registered
    combined       — same as live_preview plus apply_teacache wrapper

Pre-reqs (run once per release-train):
    scripts/_capture_latent.py --variant flux1-dev
    scripts/_capture_latent.py --variant flux2-klein-base-4b

Usage:
    uv run python scripts/run_showcase.py --scenario all --reps 3 \\
        --report _artifacts/showcase_report.json
"""

from __future__ import annotations

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

from mlx_taef.errors import FixtureLatentMissingError, SchemaVersionError  # noqa: E402

logger = logging.getLogger("mlx_taef.showcase")


SCHEMA_VERSION = 1


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
    return parser


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
    _check_latent_sha(latent)
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
    )
    vae_result = _run_orchestrator(
        latent_path=latent,
        condition="vanilla_vae",
        reps=vae_reps,
        save_dir=save_dir / "vae",
        flux_variant=flux_variant,
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


def _run_live_preview(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one FLUX.2 image, save a numbered preview frame at every step."""
    return _live_generation(args=args, with_teacache=False, scenario_dir="live_preview")


def _run_combined(args: argparse.Namespace) -> dict[str, Any]:
    """Generate one FLUX.2 image with apply_teacache + LivePreviewCallback."""
    return _live_generation(args=args, with_teacache=True, scenario_dir="combined")


class _GalleryPreviewCallback:
    """In-loop callback that saves a TAEF2 preview every step with sequence number.

    Closely mirrors LivePreviewCallback but writes numbered PNG files instead
    of overwriting a single path — so the final gallery shows progression.
    """

    def __init__(
        self,
        *,
        flux: object,
        save_dir: Path,
        prefix: str,
        latent_height: int,
        latent_width: int,
    ) -> None:
        from mlx_taef.api import TAEF2
        from mlx_taef.integrations.mflux import _try_extract_bn

        self.taef = TAEF2.from_pretrained(include_encoder=False)
        bn_mean, bn_var = _try_extract_bn(flux)
        if bn_mean is None or bn_var is None:
            raise RuntimeError(
                "could not extract Flux2VAE BN stats from the flux instance — "
                "preview gallery would be color-shifted."
            )
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.saved_paths: list[Path] = []
        self._iter = 0

    def call_in_loop(
        self,
        t: object,
        seed: object,
        prompt: object,
        latents: object,
        config: object,
        time_steps: object,
    ) -> None:
        import numpy as np
        from PIL import Image

        from mlx_taef.integrations.mflux import unpack_flux2_latent

        unpacked = unpack_flux2_latent(
            latents,  # type: ignore[arg-type]
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            bn_mean=self.bn_mean,
            bn_var=self.bn_var,
        )
        img = self.taef.decode_image(unpacked)
        target = self.save_dir / f"{self.prefix}_step{self._iter:02d}.webp"
        Image.fromarray(np.array(img[0])).save(target, "WEBP", quality=92)
        self.saved_paths.append(target)
        self._iter += 1


def _live_generation(
    *, args: argparse.Namespace, with_teacache: bool, scenario_dir: str
) -> dict[str, Any]:
    """Run one FLUX.2 generation with preview gallery + optional TeaCache wrap."""
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    save_dir = _ARTIFACTS_DIR / scenario_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Memory cap — live generation needs the full Flux2Klein 4B in memory.
    # Use the device-aware helper so this works on smaller CI runners too.
    from mlx_taef._memory_caps import install_memory_caps

    install_memory_caps()
    mx.reset_peak_memory()

    height = 512
    width = 512
    num_steps = 4
    guidance = 1.0
    seed = 42
    prompt = "a red apple on a wooden table"

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())

    teacache_stats: dict[str, Any] | None = None
    handle = None
    if with_teacache:
        from mlx_teacache import apply_teacache

        handle = apply_teacache(flux)

    callback = _GalleryPreviewCallback(
        flux=flux,
        save_dir=save_dir,
        prefix=scenario_dir,
        latent_height=height // 16,
        latent_width=width // 16,
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
    "live_preview": _run_live_preview,
    "combined": _run_combined,
}


# ---------------------------------------------------------------------------
# JSON I/O (testable in isolation)
# ---------------------------------------------------------------------------


def _build_hardware_metadata() -> dict[str, Any]:
    """Collect hardware + version metadata for the report header."""

    def _safe_version(pkg: str) -> str | None:
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "chip": platform.processor() or "unknown",  # filled by user / CI if specific
        "ram_gb": _detect_ram_gb(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "mlx_taef_version": _safe_version("mlx-taef") or "0.0.0+dev",
        "mlx_teacache_version": _safe_version("mlx-teacache"),  # may be None
        "mflux_version": _safe_version("mflux") or "unknown",
        "quantize": 4,
        "dtype": "bf16",
    }


def _detect_ram_gb() -> int:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3)
    except Exception:  # pragma: no cover
        return 0


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


def _check_latent_sha(latent: Path) -> None:
    """Verify the .sha256 sidecar matches. Warn on mismatch; raise if missing."""
    if not latent.exists():
        raise FixtureLatentMissingError(
            f"showcase latent missing at {latent}; run "
            f"`scripts/_capture_latent.py --variant <name>` first."
        )
    sidecar = latent.with_suffix(latent.suffix + ".sha256")
    if not sidecar.exists():
        logger.warning("no .sha256 sidecar at %s; skipping integrity check", sidecar)
        return
    expected = sidecar.read_text().split()[0]
    actual = hashlib.sha256(latent.read_bytes()).hexdigest()
    if expected != actual:
        logger.warning(
            "sha mismatch for %s: expected %s, got %s. Latent may have been regenerated.",
            latent.name,
            expected[:12],
            actual[:12],
        )


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

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "hardware": _build_hardware_metadata(),
        "isolation": "subprocess-per-rep",
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
