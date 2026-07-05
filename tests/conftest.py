"""Shared pytest fixtures + session-level MLX memory caps."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


# Install MLX memory caps at module-import time (NOT pytest_configure) so
# the cap lands before pytest's collection imports any worker module.
# Mirrors mlx-teacache v0.6.0 conftest.py:27-59 pattern. Prevents the kernel
# watchdog panic that unbounded wired (GPU-pinned) allocations trigger on 32 GB
# M-series Macs when a misrouted parity test loads a large model.
#
# The actual (wired_gb, memory_gb) installed is hardware-dependent: on a
# 32 GB M1 Max it lands at (20, 22); on smaller CI runners
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


# Collection gating: keep a bare `pytest` fast and offline. Tests carrying these
# markers do real I/O (network) or minutes of perf measurement (benchmark), so a
# developer running `uv run pytest` with no flags should not pay for them — they
# are skipped unless the matching opt-in flag is passed. Each tuple is
# (marker, opt-in flag, short description).
GATED_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("network", "--run-network", "real HF downloads"),
    ("benchmark", "--run-benchmark", "perf timings, slow and noisy"),
)


def _markers_to_skip(enabled_flags: set[str]) -> list[tuple[str, str]]:
    """Pure decision: which (marker, skip-reason) pairs to skip given opt-in flags.

    Separated from the pytest hook so the gating policy is unit-testable without
    a live pytest session.

    Args:
        enabled_flags: the opt-in flags present on this invocation (e.g.
            ``{"--run-network"}``).

    Returns:
        ``(marker, reason)`` pairs whose tests should be skipped this run.
    """
    return [
        (marker, f"requires {flag} ({desc})")
        for marker, flag, desc in GATED_MARKERS
        if flag not in enabled_flags
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in flags for the gated markers."""
    for marker, flag, desc in GATED_MARKERS:
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run `{marker}`-marked tests ({desc}); skipped by default",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip gated tests unless their opt-in flag was passed, so a bare run is fast+offline."""
    enabled = {flag for _marker, flag, _desc in GATED_MARKERS if config.getoption(flag)}
    for marker, reason in _markers_to_skip(enabled):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


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
