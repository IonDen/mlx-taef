"""Tests for `mlx_taef._memory_caps` — the device-aware cap clamper.

The helper exists so CLAUDE.md's 20 GB / 22 GB user-machine targets
don't crash on smaller-ceiling hardware (CI runners, 8 GB Mac mini).
Tests cover the happy path (real device) and the clamping branches
via `mx.device_info` patching.
"""

from __future__ import annotations

from unittest.mock import patch

from mlx_taef._memory_caps import (
    DESIRED_MEMORY_GB,
    DESIRED_WIRED_GB,
    compute_safe_caps_gb,
)


def test_compute_safe_caps_returns_positive_pair_on_real_device() -> None:
    wired_gb, memory_gb = compute_safe_caps_gb()
    # Apple Silicon always reports a working-set size; both should be >0.
    assert wired_gb > 0
    assert memory_gb > wired_gb


def test_compute_safe_caps_respects_desired_targets_on_large_device() -> None:
    """A device with 40 GB recommended working set returns the desired targets."""
    huge = 40 * 1024**3
    with patch(
        "mlx_taef._memory_caps.mx.device_info",
        return_value={"max_recommended_working_set_size": huge},
    ):
        wired_gb, memory_gb = compute_safe_caps_gb()
    assert wired_gb == DESIRED_WIRED_GB
    assert memory_gb == DESIRED_MEMORY_GB


def test_compute_safe_caps_clamps_below_small_ceiling() -> None:
    """An 8 GB recommended working set yields a wired cap << 20 GB."""
    eight = 8 * 1024**3
    with patch(
        "mlx_taef._memory_caps.mx.device_info",
        return_value={"max_recommended_working_set_size": eight},
    ):
        wired_gb, memory_gb = compute_safe_caps_gb()
    assert wired_gb <= 8
    assert wired_gb < DESIRED_WIRED_GB
    assert memory_gb > wired_gb
    assert memory_gb <= 8


def test_compute_safe_caps_returns_zero_on_no_working_set_size() -> None:
    """Missing key (older MLX / non-Metal) signals 'no cap available'."""
    with patch("mlx_taef._memory_caps.mx.device_info", return_value={}):
        result = compute_safe_caps_gb()
    assert result == (0, 0)


def test_compute_safe_caps_returns_zero_on_negative_size() -> None:
    with patch(
        "mlx_taef._memory_caps.mx.device_info",
        return_value={"max_recommended_working_set_size": -1},
    ):
        result = compute_safe_caps_gb()
    assert result == (0, 0)
