"""Prove the REAL collection-gating hook (not just _markers_to_skip) wires up."""

import pytest


def test_real_gating_hook_skips_benchmark_without_flag(pytester: pytest.Pytester) -> None:
    # Reuse the actual conftest hooks so this tests the real wiring.
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_collection_modifyitems"
    )
    pytester.makepyfile(
        test_inner="""
        import pytest

        @pytest.mark.benchmark
        def test_benchmark_marked():
            assert True  # body irrelevant; only skip/pass counts matter

        @pytest.mark.network
        def test_network_marked():
            assert True  # body irrelevant; only skip/pass counts matter

        def test_unmarked():
            assert True  # body irrelevant; only skip/pass counts matter
        """
    )
    # No flags: both gated tests skipped, the plain one runs.
    pytester.runpytest().assert_outcomes(skipped=2, passed=1)
    # Opt in to both gates: all three run.
    pytester.runpytest("--run-benchmark", "--run-network").assert_outcomes(passed=3)
