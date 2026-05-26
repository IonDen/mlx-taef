"""Tests for scripts/diff_showcase_report.py — pure data + math."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _make_report(median_seconds: float, ssim_median: float) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "scenarios": {
            "taef2_vs_vae": {
                "conditions": {
                    "taef2": {"median_seconds": median_seconds, "median_peak_memory_gb": 1.5},
                },
                "perceptual": {"ssim_median": ssim_median, "threshold": 0.75},
            },
        },
    }


def test_clean_diff_reports_no_regression(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_report(
        median_seconds=0.105, ssim_median=0.84
    )  # 5% wall-clock drift, 0.01 SSIM drop

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    assert regressions == []


def test_wallclock_drift_above_threshold_flagged(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_report(median_seconds=0.115, ssim_median=0.85)  # 15% wall-clock drift

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    assert len(regressions) == 1
    assert "wallclock" in regressions[0]["kind"]


def test_ssim_drop_above_threshold_flagged(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_report(median_seconds=0.10, ssim_median=0.79)  # 0.06 SSIM drop

    regressions = diff_reports(old, new, wallclock_tolerance=0.10, ssim_tolerance=0.05)
    assert len(regressions) == 1
    assert "ssim" in regressions[0]["kind"]


def test_main_exit_zero_on_clean_diff(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import main

    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(_make_report(0.1, 0.85)))
    new_path.write_text(json.dumps(_make_report(0.105, 0.84)))

    assert main([str(old_path), str(new_path)]) == 0


def test_main_exit_nonzero_on_regression(tmp_path: Path) -> None:
    from scripts.diff_showcase_report import main

    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(_make_report(0.1, 0.85)))
    new_path.write_text(json.dumps(_make_report(0.2, 0.85)))  # 100% slower

    assert main([str(old_path), str(new_path)]) == 1
