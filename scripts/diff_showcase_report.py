r"""Machine-enforced regression checker for two showcase reports.

Exits non-zero if any condition's median wall-clock drifts more than
the tolerance or SSIM drops more than the tolerance.

Usage:
    uv run python scripts/diff_showcase_report.py old.json new.json
    uv run python scripts/diff_showcase_report.py old.json new.json \
        --wallclock-tolerance 0.10 --ssim-tolerance 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def diff_reports(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    wallclock_tolerance: float = 0.10,
    ssim_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Return a list of regression records; empty list means no regression."""
    regressions: list[dict[str, Any]] = []
    old_scenarios = old.get("scenarios", {})
    new_scenarios = new.get("scenarios", {})
    for scenario, new_data in new_scenarios.items():
        old_data = old_scenarios.get(scenario, {})

        # Wall-clock per condition
        old_conditions = old_data.get("conditions", {})
        new_conditions = new_data.get("conditions", {})
        for cond, new_cond in new_conditions.items():
            old_cond = old_conditions.get(cond, {})
            old_med = old_cond.get("median_seconds")
            new_med = new_cond.get("median_seconds")
            if old_med is None or new_med is None:
                continue
            drift = (new_med - old_med) / old_med
            if drift > wallclock_tolerance:
                regressions.append({
                    "kind": "wallclock-drift",
                    "scenario": scenario,
                    "condition": cond,
                    "old_seconds": old_med,
                    "new_seconds": new_med,
                    "drift_pct": drift * 100,
                })

        # SSIM
        old_perc = old_data.get("perceptual", {})
        new_perc = new_data.get("perceptual", {})
        old_ssim = old_perc.get("ssim_median")
        new_ssim = new_perc.get("ssim_median")
        if (
            old_ssim is not None
            and new_ssim is not None
            and (old_ssim - new_ssim) > ssim_tolerance
        ):
            regressions.append({
                "kind": "ssim-drop",
                "scenario": scenario,
                "old_ssim": old_ssim,
                "new_ssim": new_ssim,
                "drop": old_ssim - new_ssim,
            })

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
        old, new,
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
