"""Subprocess-per-rep decoder bench worker + orchestrator.

Reused by `scripts/run_showcase.py` for the `taef*_vs_vae` scenarios.
Standalone CLI for regression-tracking.

Sentinel contract (deliberate tightening of mlx-teacache template):
- `::BENCH_RESULT::<json>` is line-start, exactly one per worker,
  JSON one-liner.
- Multiple sentinels in one worker's stdout raise TaefError.
- Missing sentinel raises TaefError.

See docs/superpowers/specs/2026-05-26-mlx-taef-v0.2.0-design.md
Section 3 for the full design.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlx_taef.errors import TaefError
from mlx_taef.variants import get_memory_cap_hint
from scripts._caps import FULL_VAE_CAP_GB

SENTINEL_PREFIX = "::BENCH_RESULT::"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True, help="Input safetensors latent.")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["taef1", "taef2", "vanilla_vae"],
        help="Which decoder to run.",
    )
    parser.add_argument(
        "--flux-variant",
        default="flux2-klein-base-4b",
        choices=list(FULL_VAE_CAP_GB.keys()),
        help="Which full Flux variant the vanilla_vae condition uses.",
    )
    parser.add_argument("--reps", type=int, default=5, help="Reps for the orchestrator (default 5 for taef, 3 for full VAE).")
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
    sentinel_lines = [
        line for line in stdout.splitlines() if line.startswith(SENTINEL_PREFIX)
    ]
    if len(sentinel_lines) == 0:
        raise TaefError(f"no sentinel found in worker stdout (stdout: {stdout[:500]!r})")
    if len(sentinel_lines) > 1:
        raise TaefError(
            f"multiple sentinels in worker stdout (got {len(sentinel_lines)}, expected 1)"
        )
    payload_str = sentinel_lines[0][len(SENTINEL_PREFIX):]
    payload: dict[str, Any] = json.loads(payload_str)
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
        "--latent", str(latent_path),
        "--condition", condition,
        "--flux-variant", flux_variant,
        "--rep", str(rep),
        "--save-to", str(save_to),
    ]
    if cap_gb is not None:
        cmd.extend(["--applied-cap-gb", str(cap_gb)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        return {"condition": condition, "rep": rep, "status": "failed", "error": "timeout"}

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
) -> dict[str, Any]:
    """Run all reps for one condition; aggregate."""
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
    return {
        "condition": condition,
        "applied_cap_gb": cap_gb,
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


def _worker_main(args: argparse.Namespace) -> int:
    """Run one (condition, rep) inside this subprocess. Emit sentinel."""
    # Heavy MLX path — implemented when the user runs the bench. For TDD
    # purposes the orchestrator's correctness is tested via mocked _run_one_rep.
    raise NotImplementedError(
        "Worker MLX implementation lands in the bench-day commit; "
        "the orchestrator + sentinel contract is implemented + tested today."
    )


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
