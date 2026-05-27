"""Shared pytest fixtures + session-level MLX memory caps."""

from __future__ import annotations

from pathlib import Path

import pytest


# Install MLX memory caps at module-import time (NOT pytest_configure) so
# the cap lands before pytest's collection imports any worker module.
# Mirrors mlx-teacache v0.6.0 conftest.py:27-59 pattern. Prevents the kernel
# watchdog panic documented in CLAUDE.md "Memory guardrails" rule on 32 GB
# M-series Macs when a misrouted parity test loads a large model.
#
# The actual (wired_gb, memory_gb) installed is hardware-dependent: on a
# 32 GB M1 Max it lands at (20, 22) per CLAUDE.md; on smaller CI runners
# the helper clamps below the device's max_recommended_working_set_size.
def _install_mlx_memory_caps() -> tuple[int, int]:
    try:
        from mlx_taef._memory_caps import install_memory_caps
    except ImportError:  # pragma: no cover - MLX always present on Apple Silicon
        return (0, 0)
    try:
        return install_memory_caps()
    except Exception:  # pragma: no cover - older MLX / non-Metal env
        return (0, 0)


INSTALLED_CAPS_GB = _install_mlx_memory_caps()


CONVERTED_DIR = Path(__file__).parent / "converted"
REF_DIR = Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def converted_dir() -> Path:
    """Path to pre-converted MLX safetensors files (committed in repo)."""
    return CONVERTED_DIR


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    """Path to reference fixtures (PyTorch-decoded outputs, committed in repo)."""
    return REF_DIR
