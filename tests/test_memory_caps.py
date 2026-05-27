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


def test_install_memory_caps_does_not_raise_on_small_ceiling() -> None:
    """Regression: the 2026-05-27 CI failure was install_memory_caps
    raising ValueError when mx.set_wired_limit was handed a value
    exceeding max_recommended_working_set_size. Patch device_info to
    8 GB and confirm install_memory_caps completes cleanly."""
    from mlx_taef._memory_caps import install_memory_caps

    eight = 8 * 1024**3
    # Patch both device_info (driving the clamp) AND set_wired_limit
    # so the test doesn't actually mess with this process's MLX state.
    with (
        patch(
            "mlx_taef._memory_caps.mx.device_info",
            return_value={"max_recommended_working_set_size": eight},
        ),
        patch("mlx_taef._memory_caps.mx.set_wired_limit") as set_wired,
        patch("mlx_taef._memory_caps.mx.set_memory_limit") as set_mem,
    ):
        wired_gb, memory_gb = install_memory_caps()
    assert wired_gb > 0
    assert wired_gb <= 8
    assert memory_gb > wired_gb
    set_wired.assert_called_once()
    set_mem.assert_called_once()
    # The bytes passed must be strictly < the 8 GB ceiling (else the
    # real set_wired_limit would have raised).
    (called_wired_bytes,) = set_wired.call_args.args
    assert called_wired_bytes < eight


def test_install_memory_caps_no_op_on_non_metal_env() -> None:
    """When device_info returns nothing useful, install_memory_caps
    returns (0, 0) without touching MLX state."""
    from mlx_taef._memory_caps import install_memory_caps

    with (
        patch("mlx_taef._memory_caps.mx.device_info", return_value={}),
        patch("mlx_taef._memory_caps.mx.set_wired_limit") as set_wired,
    ):
        result = install_memory_caps()
    assert result == (0, 0)
    set_wired.assert_not_called()
