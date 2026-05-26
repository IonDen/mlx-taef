"""Shared pytest fixtures + session-level MLX memory caps."""

from __future__ import annotations

from pathlib import Path

import pytest


# Install MLX memory caps at module-import time (NOT pytest_configure) so
# the cap lands before pytest's collection imports any worker module.
# Mirrors mlx-teacache v0.6.0 conftest.py:27-59 pattern. Prevents the kernel
# watchdog panic documented in CLAUDE.md "Memory guardrails" rule on 32 GB
# M-series Macs when a misrouted parity test loads a large model.
def _install_mlx_memory_caps() -> None:
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - MLX always present on Apple Silicon
        return
    try:
        mx.set_wired_limit(20 * 1024**3)
        mx.set_memory_limit(22 * 1024**3)
    except Exception:  # pragma: no cover - older MLX / non-Metal env
        return


_install_mlx_memory_caps()


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
