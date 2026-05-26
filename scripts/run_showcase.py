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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from mlx_taef.errors import FixtureLatentMissingError, SchemaVersionError

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

def _run_taef2_vs_vae(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Heavy MLX path — lands in the bench-day commit.")


def _run_taef1_vs_vae(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Heavy MLX path — lands in the bench-day commit.")


def _run_live_preview(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Heavy MLX path — lands in the bench-day commit.")


def _run_combined(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Heavy MLX path — lands in the bench-day commit.")


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
            f"unknown schema_version: got {data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
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
            latent.name, expected[:12], actual[:12],
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
            score = float(structural_similarity(
                ref_img, cand_img, channel_axis=-1, data_range=1.0,
            ))
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
