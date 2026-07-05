"""Tests for scripts/diff_showcase_report.py — pure data + math.

Fixtures now match the ACTUAL schema produced by run_showcase.py (no
`conditions` / `perceptual` sub-keys; condition data lives directly
under the scenario as `taef` / `vanilla_vae`, and `ssim_median` lives
at scenario level). The original tests used a sketch schema that
diverged from real output — the regression bug it masked is the reason
this file was rewritten.
"""

import json
from pathlib import Path
from typing import Any


def _make_vs_vae_report(
    median_seconds: float,
    ssim_median: float,
    *,
    taef_peak_gb: float = 0.59,
    vae_peak_gb: float = 2.37,
) -> dict[str, Any]:
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
                    "median_peak_memory_gb": taef_peak_gb,
                },
                "vanilla_vae": {
                    "condition": "vanilla_vae",
                    "median_seconds": median_seconds * 8,  # vae is slower
                    "median_peak_memory_gb": vae_peak_gb,
                },
                "ssim_per_pair": [ssim_median],
                "ssim_median": ssim_median,
            },
        },
    }


def _make_live_report(elapsed_s: float, *, peak_memory_gb: float = 10.0) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "scenarios": {
            "live_preview": {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "peak_memory_gb": peak_memory_gb,
            },
        },
    }


def _make_combined_report(
    *,
    elapsed_s: float = 8.0,
    peak_memory_gb: float = 7.9,
    skipped_count: int = 1,
) -> dict[str, Any]:
    """Build a report shaped like run_showcase.py's `combined` scenario,
    which carries the TeaCache `skipped_count` the integration headline depends on."""
    return {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "scenarios": {
            "combined": {
                "status": "ok",
                "elapsed_s": elapsed_s,
                "peak_memory_gb": peak_memory_gb,
                "teacache": {"skipped_count": skipped_count},
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


# --- peak memory (the headline 5-7x memory win — must be guarded) ---


def test_peak_memory_drift_above_threshold_flagged_for_taef() -> None:
    """taef's peak decode memory creeping up past tolerance is a regression of
    the library's headline claim, so the differ must flag it."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85, taef_peak_gb=0.59)
    new = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85, taef_peak_gb=0.59 * 1.2)

    regressions = diff_reports(old, new, memory_tolerance=0.10)
    flagged = [
        r for r in regressions if r["kind"] == "peak-memory-drift" and r["condition"] == "taef"
    ]
    assert len(flagged) == 1, f"expected a peak-memory-drift on taef, got {regressions}"


def test_peak_memory_within_tolerance_reports_no_regression() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85, taef_peak_gb=0.59)
    new = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85, taef_peak_gb=0.59 * 1.05)

    regressions = diff_reports(old, new, memory_tolerance=0.10)
    assert [r for r in regressions if r["kind"] == "peak-memory-drift"] == []


def test_peak_memory_drift_flagged_for_live_scenario() -> None:
    """Live/combined scenarios carry scenario-level peak_memory_gb."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_live_report(elapsed_s=10.0, peak_memory_gb=10.0)
    new = _make_live_report(elapsed_s=10.0, peak_memory_gb=12.0)  # 20% up

    regressions = diff_reports(old, new, memory_tolerance=0.10)
    flagged = [r for r in regressions if r["kind"] == "peak-memory-drift"]
    assert len(flagged) == 1
    assert flagged[0]["scenario"] == "live_preview"
    assert flagged[0]["condition"] == "(scenario)"


# --- TeaCache skipped_count floor (combined integration must keep skipping) ---


def test_skipped_count_drop_to_zero_flagged_for_combined() -> None:
    """A TeaCache integration regression that stops skipping steps drops
    combined.teacache.skipped_count to 0; the differ must flag it."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_combined_report(skipped_count=1)
    new = _make_combined_report(skipped_count=0)

    regressions = diff_reports(old, new)
    flagged = [r for r in regressions if r["kind"] == "skipped-count-drop"]
    assert len(flagged) == 1, f"expected a skipped-count-drop, got {regressions}"
    assert flagged[0]["scenario"] == "combined"


def test_skipped_count_held_or_increased_reports_no_regression() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_combined_report(skipped_count=1)
    for new_count in (1, 2):
        new = _make_combined_report(skipped_count=new_count)
        regressions = diff_reports(old, new)
        assert [r for r in regressions if r["kind"] == "skipped-count-drop"] == []


def test_self_diff_with_skewed_peak_memory_finds_regression() -> None:
    """Integration: perturb the committed report's taef peak memory and verify
    the differ catches it (proves it inspects the real median_peak_memory_gb)."""
    from scripts.diff_showcase_report import diff_reports

    report_path = Path(__file__).parent.parent / "_artifacts" / "showcase_report.json"
    if not report_path.exists():
        import pytest

        pytest.skip(f"no committed showcase report at {report_path}")

    old = json.loads(report_path.read_text())
    new = json.loads(report_path.read_text())
    new["scenarios"]["taef2_vs_vae"]["taef"]["median_peak_memory_gb"] *= 1.5
    regressions = diff_reports(old, new)
    flagged = [r for r in regressions if r["kind"] == "peak-memory-drift"]
    assert flagged, f"expected a peak-memory regression, got {regressions}"


def test_self_diff_with_zeroed_skipped_count_finds_regression() -> None:
    """Integration: zero out the committed report's combined skipped_count and
    verify the differ flags the TeaCache integration regression."""
    from scripts.diff_showcase_report import diff_reports

    report_path = Path(__file__).parent.parent / "_artifacts" / "showcase_report.json"
    if not report_path.exists():
        import pytest

        pytest.skip(f"no committed showcase report at {report_path}")

    old = json.loads(report_path.read_text())
    new = json.loads(report_path.read_text())
    new["scenarios"]["combined"]["teacache"]["skipped_count"] = 0
    regressions = diff_reports(old, new)
    flagged = [r for r in regressions if r["kind"] == "skipped-count-drop"]
    assert flagged, f"expected a skipped-count-drop, got {regressions}"


# --- the metric/block disappearing entirely is the worst regression and must
#     also be flagged (not just a worse number) ---


def test_skipped_count_block_removed_flagged() -> None:
    """Losing the whole teacache block (the TeaCache wiring removed outright) is
    a worse regression than 1 -> 0, and must not pass silently."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_combined_report(skipped_count=1)
    new = _make_combined_report(skipped_count=1)
    del new["scenarios"]["combined"]["teacache"]

    regressions = diff_reports(old, new)
    flagged = [r for r in regressions if r["kind"] == "skipped-count-missing"]
    assert len(flagged) == 1, f"expected a skipped-count-missing, got {regressions}"
    assert flagged[0]["scenario"] == "combined"


def test_peak_memory_field_removed_flagged_for_condition() -> None:
    """A baseline that reports median_peak_memory_gb against a new report that
    dropped it should fail loud (the headline metric vanished)."""
    from scripts.diff_showcase_report import diff_reports

    old = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85)
    new = _make_vs_vae_report(median_seconds=0.10, ssim_median=0.85)
    del new["scenarios"]["taef2_vs_vae"]["taef"]["median_peak_memory_gb"]

    regressions = diff_reports(old, new)
    flagged = [
        r for r in regressions if r["kind"] == "peak-memory-missing" and r["condition"] == "taef"
    ]
    assert len(flagged) == 1, f"expected a peak-memory-missing on taef, got {regressions}"


def test_peak_memory_field_removed_flagged_for_scenario() -> None:
    from scripts.diff_showcase_report import diff_reports

    old = _make_live_report(elapsed_s=10.0)
    new = _make_live_report(elapsed_s=10.0)
    del new["scenarios"]["live_preview"]["peak_memory_gb"]

    regressions = diff_reports(old, new)
    flagged = [r for r in regressions if r["kind"] == "peak-memory-missing"]
    assert len(flagged) == 1
    assert flagged[0]["scenario"] == "live_preview"
    assert flagged[0]["condition"] == "(scenario)"


def test_whole_scenario_missing_from_new_report_is_flagged() -> None:
    """A scenario present in the baseline but absent from the new report must be flagged —
    otherwise a partial re-run silently passes as 'no regressions'."""
    from scripts.diff_showcase_report import diff_reports

    old = {"scenarios": {"taef1_vs_vae": {"ssim_median": 0.95}, "live_preview": {"elapsed_s": 2.0}}}
    new = {"scenarios": {"taef1_vs_vae": {"ssim_median": 0.95}}}  # live_preview dropped
    regs = diff_reports(old, new)
    kinds = {(r["kind"], r["scenario"]) for r in regs}
    assert ("scenario-missing", "live_preview") in kinds


def test_ssim_median_missing_from_new_report_is_flagged() -> None:
    """A dropped ssim_median (baseline had it, new omits it) must be flagged, matching the
    existing peak-memory-missing guard."""
    from scripts.diff_showcase_report import diff_reports

    old = {"scenarios": {"taef1_vs_vae": {"ssim_median": 0.95}}}
    new = {"scenarios": {"taef1_vs_vae": {}}}  # ssim_median gone
    regs = diff_reports(old, new)
    assert any(r["kind"] == "ssim-missing" for r in regs)


def test_condition_median_seconds_missing_from_new_report_is_flagged() -> None:
    """A dropped per-condition median_seconds (baseline had it, new omits it) must flag —
    otherwise a latency field can vanish and still pass as 'no regressions'."""
    from scripts.diff_showcase_report import diff_reports

    old = {"scenarios": {"taef1_vs_vae": {"taef": {"median_seconds": 0.18}}}}
    new = {"scenarios": {"taef1_vs_vae": {"taef": {}}}}  # median_seconds gone
    regs = diff_reports(old, new)
    assert any(r["kind"] == "wallclock-missing" for r in regs)


def test_scenario_elapsed_s_missing_from_new_report_is_flagged() -> None:
    """A dropped scenario-level elapsed_s (live_preview/combined) must flag too."""
    from scripts.diff_showcase_report import diff_reports

    old = {"scenarios": {"live_preview": {"elapsed_s": 2.0}}}
    new = {"scenarios": {"live_preview": {}}}  # elapsed_s gone
    regs = diff_reports(old, new)
    assert any(r["kind"] == "wallclock-missing" and r["condition"] == "(scenario)" for r in regs)
