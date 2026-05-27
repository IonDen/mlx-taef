"""Tests for scripts/diff_showcase_report.py — pure data + math.

Fixtures now match the ACTUAL schema produced by run_showcase.py (no
`conditions` / `perceptual` sub-keys; condition data lives directly
under the scenario as `taef` / `vanilla_vae`, and `ssim_median` lives
at scenario level). The original tests used a sketch schema that
diverged from real output — the regression bug it masked is the reason
this file was rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_vs_vae_report(median_seconds: float, ssim_median: float) -> dict[str, Any]:
    """Build a report shaped like run_showcase.py's `taef2_vs_vae` output."""
    return {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "scenarios": {
            "taef2_vs_vae": {
                "status": "ok",
                "taef": {
                    "condition": "taef2",
                    "median_seconds": median_seconds,
                    "median_peak_memory_gb": 0.59,
                },
                "vanilla_vae": {
                    "condition": "vanilla_vae",
                    "median_seconds": median_seconds * 8,  # vae is slower
                    "median_peak_memory_gb": 2.37,
                },
                "ssim_per_pair": [ssim_median],
                "ssim_median": ssim_median,
            },
        },
    }


def _make_live_report(elapsed_s: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "scenarios": {
            "live_preview": {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "peak_memory_gb": 10.0,
            },
        },
    }


def test_clean_diff_reports_no_regression() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_vs_vae_report(median_seconds=0.105, ssim_median=0.84)

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    assert regressions == []


def test_wallclock_drift_above_threshold_flagged_for_taef() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_vs_vae_report(median_seconds=0.115, ssim_median=0.85)

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    flagged = [r for r in regressions if r["condition"] == "taef"]
    assert len(flagged) == 1
    assert flagged[0]["kind"] == "wallclock-drift"


def test_wallclock_drift_flagged_for_live_scenario() -> None:
    """Live scenarios use scenario-level elapsed_s, not a nested condition."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_live_report(elapsed_s=10.0)
    new = _make_live_report(elapsed_s=11.5)  # 15% drift

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    assert len(regressions) == 1
    assert regressions[0]["kind"] == "wallclock-drift"
    assert regressions[0]["scenario"] == "live_preview"
    assert regressions[0]["condition"] == "(scenario)"


def test_ssim_drop_above_threshold_flagged() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.79)

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    ssim_regs = [r for r in regressions if r["kind"] == "ssim-drop"]
    assert len(ssim_regs) == 1


def test_main_exit_zero_on_clean_diff(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import main

    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(_make_vs_vae_report(0.1, 0.85)))
    new_path.write_text(json.dumps(_make_vs_vae_report(0.105, 0.84)))

    assert main([str(old_path), str(new_path)]) == 0


def test_main_exit_nonzero_on_regression(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import main

    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(_make_vs_vae_report(0.1, 0.85)))
    new_path.write_text(json.dumps(_make_vs_vae_report(0.2, 0.85)))  # 100% slower

    assert main([str(old_path), str(new_path)]) == 1


def test_committed_showcase_report_against_itself_finds_no_regression() -> None:
    """Integration: the committed _artifacts/showcase_report.json diffed
    against itself MUST report no regressions. This is the test that
    would have caught the schema mismatch the original implementation
    silently absorbed."""
    from scripts.diff_showcase_report import diff_reports

    report_path = Path(__file__).parent.parent / "_artifacts" / "showcase_report.json"
    if not report_path.exists():
        # Pre-bench environments — skip without failing.
        import pytest

        pytest.skip(f"no committed showcase report at {report_path}")

    report = json.loads(report_path.read_text())
    assert diff_reports(report, report) == []


def test_self_diff_with_skewed_taef_value_finds_regression() -> None:
    """Integration: copy the committed report, perturb one taef
    `median_seconds` by >10%, and verify the regression checker catches
    it. This proves diff_reports actually inspects the real schema."""
    from scripts.diff_showcase_report import diff_reports

    report_path = Path(__file__).parent.parent / "_artifacts" / "showcase_report.json"
    if not report_path.exists():
        import pytest

        pytest.skip(f"no committed showcase report at {report_path}")

    old = json.loads(report_path.read_text())
    new = json.loads(report_path.read_text())
    # Bump taef2's median_seconds by 50% in the new report
    new["scenarios"]["taef2_vs_vae"]["taef"]["median_seconds"] *= 1.5
    regressions = diff_reports(old, new, wallclock_tolerance=0.10)
    flagged = [r for r in regressions if r.get("condition") == "taef"]
    assert flagged, f"expected a wallclock regression on taef, got {regressions}"
