r"""Machine-enforced regression checker for two showcase reports.

Exits non-zero if any wall-clock or peak-memory metric drifts more than the
tolerance, any SSIM drops more than the tolerance, the TeaCache skip count
drops, or a baseline metric/block disappears from the new report. Schema
follows the actual output of `scripts/run_showcase.py` (NOT the early-spec
sketch):

    scenarios.<name>.{taef,vanilla_vae}.median_seconds         # taef*_vs_vae
    scenarios.<name>.{taef,vanilla_vae}.median_peak_memory_gb  # taef*_vs_vae
    scenarios.<name>.ssim_median                               # taef*_vs_vae
    scenarios.<name>.elapsed_s                                 # live_preview, combined
    scenarios.<name>.peak_memory_gb                            # live_preview, combined
    scenarios.combined.teacache.skipped_count                  # combined

Usage:
    uv run python scripts/run_showcase.py --report new.json
    uv run python scripts/diff_showcase_report.py \
        _artifacts/showcase_report.json new.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Keep the documented `python scripts/diff_showcase_report.py` invocation
# working even when the caller's current directory is outside the repository.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    memory_tolerance: float = 0.10,
) -> list[dict[str, Any]]:
    """Return a list of regression records; empty list means no regression.

    Checks (per scenario):
      - For `taef*_vs_vae`: each known condition's `median_seconds` and
        `median_peak_memory_gb`, plus scenario-level `ssim_median`.
      - For `live_preview` / `combined`: scenario-level `elapsed_s` and
        `peak_memory_gb`.
      - For any scenario carrying `teacache.skipped_count` (the `combined`
        scenario): a floor — a drop below the baseline count is flagged, so a
        TeaCache integration regression that stops skipping steps cannot pass
        silently.

    Peak memory is the library's headline win (a few-MB decoder vs the full
    VAE), so a memory regression is gated with the same drift shape as
    wall-clock.
    """
    regressions: list[dict[str, Any]] = []
    old_scenarios = old.get("scenarios", {})
    new_scenarios = new.get("scenarios", {})

    for scenario in sorted(old_scenarios.keys() | new_scenarios.keys()):
        if scenario not in new_scenarios:
            regressions.append(
                {
                    "kind": "scenario-missing",
                    "scenario": scenario,
                    "detail": "present in baseline, absent from new report",
                }
            )
            continue
        new_data = new_scenarios[scenario]
        old_data = old_scenarios.get(scenario, {})

        old_preview_count = old_data.get("preview_count")
        new_preview_count = new_data.get("preview_count")
        if (
            isinstance(old_preview_count, int)
            and not isinstance(old_preview_count, bool)
            and (
                not isinstance(new_preview_count, int)
                or isinstance(new_preview_count, bool)
                or new_preview_count < old_preview_count
            )
        ):
            regressions.append(
                {
                    "kind": "preview-count-drop",
                    "scenario": scenario,
                    "old_count": old_preview_count,
                    "new_count": new_preview_count,
                }
            )

        # taef*_vs_vae conditions → median_seconds + median_peak_memory_gb per condition
        for cond in _VS_VAE_CONDITION_KEYS:
            new_cond = new_data.get(cond)
            old_cond = old_data.get(cond)
            if not isinstance(new_cond, dict) or not isinstance(old_cond, dict):
                continue
            old_med = old_cond.get("median_seconds")
            new_med = new_cond.get("median_seconds")
            if isinstance(old_med, (int, float)) and old_med > 0:
                if not isinstance(new_med, (int, float)):
                    regressions.append(
                        {
                            "kind": "wallclock-missing",
                            "scenario": scenario,
                            "condition": cond,
                            "old_seconds": old_med,
                            "new_seconds": new_med,
                        }
                    )
                else:
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
            old_mem = old_cond.get("median_peak_memory_gb")
            new_mem = new_cond.get("median_peak_memory_gb")
            if isinstance(old_mem, (int, float)) and old_mem > 0:
                if not isinstance(new_mem, (int, float)):
                    regressions.append(
                        {
                            "kind": "peak-memory-missing",
                            "scenario": scenario,
                            "condition": cond,
                            "old_gb": old_mem,
                            "new_gb": new_mem,
                        }
                    )
                else:
                    mem_drift = (new_mem - old_mem) / old_mem
                    if mem_drift > memory_tolerance:
                        regressions.append(
                            {
                                "kind": "peak-memory-drift",
                                "scenario": scenario,
                                "condition": cond,
                                "old_gb": old_mem,
                                "new_gb": new_mem,
                                "drift_pct": mem_drift * 100,
                            }
                        )

        # Live scenarios → scenario-level elapsed_s
        old_elapsed = old_data.get("elapsed_s")
        new_elapsed = new_data.get("elapsed_s")
        if isinstance(old_elapsed, (int, float)) and old_elapsed > 0:
            if not isinstance(new_elapsed, (int, float)):
                regressions.append(
                    {
                        "kind": "wallclock-missing",
                        "scenario": scenario,
                        "condition": "(scenario)",
                        "old_seconds": old_elapsed,
                        "new_seconds": new_elapsed,
                    }
                )
            else:
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

        # Scenario-level peak_memory_gb (live_preview / combined)
        old_peak = old_data.get("peak_memory_gb")
        new_peak = new_data.get("peak_memory_gb")
        if isinstance(old_peak, (int, float)) and old_peak > 0:
            if not isinstance(new_peak, (int, float)):
                regressions.append(
                    {
                        "kind": "peak-memory-missing",
                        "scenario": scenario,
                        "condition": "(scenario)",
                        "old_gb": old_peak,
                        "new_gb": new_peak,
                    }
                )
            else:
                mem_drift = (new_peak - old_peak) / old_peak
                if mem_drift > memory_tolerance:
                    regressions.append(
                        {
                            "kind": "peak-memory-drift",
                            "scenario": scenario,
                            "condition": "(scenario)",
                            "old_gb": old_peak,
                            "new_gb": new_peak,
                            "drift_pct": mem_drift * 100,
                        }
                    )

        # TeaCache skipped_count floor: a drop below the baseline count means
        # the integration stopped skipping steps (the combined scenario).
        old_skipped = (old_data.get("teacache") or {}).get("skipped_count")
        new_skipped = (new_data.get("teacache") or {}).get("skipped_count")
        if isinstance(old_skipped, int) and not isinstance(old_skipped, bool):
            if not isinstance(new_skipped, int) or isinstance(new_skipped, bool):
                # The teacache block / skipped_count field disappeared (the
                # TeaCache wiring removed outright) — worse than a 1 -> 0 drop.
                regressions.append(
                    {
                        "kind": "skipped-count-missing",
                        "scenario": scenario,
                        "old_skipped": old_skipped,
                        "new_skipped": new_skipped,
                    }
                )
            elif new_skipped < old_skipped:
                regressions.append(
                    {
                        "kind": "skipped-count-drop",
                        "scenario": scenario,
                        "old_skipped": old_skipped,
                        "new_skipped": new_skipped,
                    }
                )

        # SSIM at scenario level (taef*_vs_vae only — live scenarios have no SSIM)
        old_ssim = old_data.get("ssim_median")
        new_ssim = new_data.get("ssim_median")
        if isinstance(old_ssim, (int, float)):
            if not isinstance(new_ssim, (int, float)):
                regressions.append(
                    {
                        "kind": "ssim-missing",
                        "scenario": scenario,
                        "old_ssim": old_ssim,
                        "new_ssim": new_ssim,
                    }
                )
            elif (old_ssim - new_ssim) > ssim_tolerance:
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
    parser.add_argument("--memory-tolerance", type=float, default=0.10)
    args = parser.parse_args(argv)

    from scripts.run_showcase import _load_report

    old = _load_report(args.old)
    new = _load_report(args.new)

    regressions = diff_reports(
        old,
        new,
        wallclock_tolerance=args.wallclock_tolerance,
        ssim_tolerance=args.ssim_tolerance,
        memory_tolerance=args.memory_tolerance,
    )
    if not regressions:
        print("No regressions detected.")
        return 0
    for r in regressions:
        print(json.dumps(r, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
