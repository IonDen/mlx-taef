"""Tests for the conftest collection-gating logic (network/benchmark opt-in).

These cover the pure decision (`_markers_to_skip`) directly; the pytest wiring
(`pytest_addoption` / `pytest_collection_modifyitems`) is exercised behaviorally
by running the suite with and without the opt-in flags.
"""

from tests.conftest import _markers_to_skip


def test_no_flags_skips_network_and_benchmark() -> None:
    """A bare run (no opt-in flags present) skips both gated markers."""
    skipped = {marker for marker, _reason in _markers_to_skip(set())}
    assert skipped == {"network", "benchmark"}


def test_run_network_opts_in_network_only() -> None:
    """--run-network un-skips network but leaves benchmark skipped."""
    skipped = {marker for marker, _reason in _markers_to_skip({"--run-network"})}
    assert skipped == {"benchmark"}


def test_run_benchmark_opts_in_benchmark_only() -> None:
    """--run-benchmark un-skips benchmark but leaves network skipped."""
    skipped = {marker for marker, _reason in _markers_to_skip({"--run-benchmark"})}
    assert skipped == {"network"}


def test_all_flags_skip_nothing() -> None:
    """With both opt-in flags present, nothing is gated."""
    assert _markers_to_skip({"--run-network", "--run-benchmark"}) == []
