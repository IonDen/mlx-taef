r"""Machine-enforced regression checker for two showcase reports.

Exits non-zero if any wall-clock metric drifts more than the tolerance
or any SSIM drops more than the tolerance. Schema follows the actual
output of `scripts/run_showcase.py` (NOT the early-spec sketch):

    scenarios.<name>.{taef,vanilla_vae}.median_seconds  # taef*_vs_vae
    scenarios.<name>.ssim_median                        # taef*_vs_vae
    scenarios.<name>.elapsed_s                          # live_preview, combined

Usage:
    uv run python scripts/run_showcase.py --report new.json
    uv run python scripts/diff_showcase_report.py \
        _artifacts/showcase_report.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Condition keys inside a `taef*_vs_vae` scenario whose `median_seconds`
# we compare. Keep this list narrow so we don't accidentally compare
# unrelated dict-shaped fields.
_VS_VAE_CONDITION_KEYS = ("taef", "vanilla_vae", "taef1", "taef2")


def diff_reports(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    wallclock_tolerance: float = 0.10,
    ssim_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Return a list of regression records; empty list means no regression.

    Checks (per scenario):
      - For `taef*_vs_vae`: each known condition's `median_seconds`, plus
        scenario-level `ssim_median`.
      - For `live_preview` / `combined`: scenario-level `elapsed_s`.
    """
    regressions: list[dict[str, Any]] = []
    old_scenarios = old.get("scenarios", {})
    new_scenarios = new.get("scenarios", {})

    for scenario, new_data in new_scenarios.items():
        old_data = old_scenarios.get(scenario, {})

        # taef*_vs_vae conditions → median_seconds per condition
        for cond in _VS_VAE_CONDITION_KEYS:
            new_cond = new_data.get(cond)
            old_cond = old_data.get(cond)
            if not isinstance(new_cond, dict) or not isinstance(old_cond, dict):
                continue
            old_med = old_cond.get("median_seconds")
            new_med = new_cond.get("median_seconds")
            if old_med is None or new_med is None or old_med <= 0:
                continue
            drift = (new_med - old_med) / old_med
            if drift > wallclock_tolerance:
                regressions.append(
                    {
                        "kind": "wallclock-drift",
                        "scenario": scenario,
                        "condition": cond,
                        "old_seconds": old_med,
                        "new_seconds": new_med,
                        "drift_pct": drift * 100,
                    }
                )

        # Live scenarios → scenario-level elapsed_s
        old_elapsed = old_data.get("elapsed_s")
        new_elapsed = new_data.get("elapsed_s")
        if (
            isinstance(old_elapsed, (int, float))
            and isinstance(new_elapsed, (int, float))
            and old_elapsed > 0
        ):
            drift = (new_elapsed - old_elapsed) / old_elapsed
            if drift > wallclock_tolerance:
                regressions.append(
                    {
                        "kind": "wallclock-drift",
                        "scenario": scenario,
                        "condition": "(scenario)",
                        "old_seconds": old_elapsed,
                        "new_seconds": new_elapsed,
                        "drift_pct": drift * 100,
                    }
                )

        # SSIM at scenario level (taef*_vs_vae only — live scenarios have no SSIM)
        old_ssim = old_data.get("ssim_median")
        new_ssim = new_data.get("ssim_median")
        if (
            isinstance(old_ssim, (int, float))
            and isinstance(new_ssim, (int, float))
            and (old_ssim - new_ssim) > ssim_tolerance
        ):
            regressions.append(
                {
                    "kind": "ssim-drop",
                    "scenario": scenario,
                    "old_ssim": old_ssim,
                    "new_ssim": new_ssim,
                    "drop": old_ssim - new_ssim,
                }
            )

    return regressions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on clean diff, 1 on regression."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--wallclock-tolerance", type=float, default=0.10)
    parser.add_argument("--ssim-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())

    regressions = diff_reports(
        old,
        new,
        wallclock_tolerance=args.wallclock_tolerance,
        ssim_tolerance=args.ssim_tolerance,
    )
    if not regressions:
        print("No regressions detected.")
        return 0
    for r in regressions:
        print(json.dumps(r, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
