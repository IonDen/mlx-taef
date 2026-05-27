"""Hardware-aware MLX memory caps (kernel-watchdog panic guard).

The user's CLAUDE.md memory-guardrails rule mandates wired + memory caps
before any heavy model load on 32 GB M-series Macs. The targets there
are (20 GB wired, 22 GB memory) — but those values exceed the system
`max_recommended_working_set_size` on smaller Apple Silicon (CI runners,
8 GB Mac mini, etc.), and `mx.set_wired_limit` raises ValueError when
asked to exceed that ceiling.

This module computes caps that fit the actual device. On a 32 GB M1 Max
(max_recommended ~25 GB) it returns (20, 22) unchanged. On a smaller
runner it clamps below `max_recommended - HEADROOM_GB`.

Refs: CLAUDE.md "Memory guardrails for heavy generations on 32 GB
unified memory"; ml-explore/mlx-lm issue #883 (wired memory is the
root cause of kernel watchdog panics on Apple Silicon).
"""

import mlx.core as mx

DESIRED_WIRED_GB = 20
DESIRED_MEMORY_GB = 22
HEADROOM_GB = 2


def compute_safe_caps_gb() -> tuple[int, int]:
    """Return (wired_gb, memory_gb) that fit the current device.

    Reads `mx.device_info()["max_recommended_working_set_size"]` and
    clamps the desired CLAUDE.md targets to fit. Returns (0, 0) when
    the device does not report a working-set size (older MLX, non-Metal
    env) — the caller should treat that as a no-op signal.
    """
    info = mx.device_info()
    max_bytes = int(info.get("max_recommended_working_set_size", 0))
    max_gb = max_bytes // (1024**3)
    if max_gb <= 0:
        return (0, 0)
    wired_gb = min(DESIRED_WIRED_GB, max(1, max_gb - HEADROOM_GB))
    memory_gb = min(DESIRED_MEMORY_GB, max(wired_gb + 1, max_gb))
    return (wired_gb, memory_gb)


def install_memory_caps() -> tuple[int, int]:
    """Apply wired + memory caps for the current device.

    Returns the `(wired_gb, memory_gb)` actually installed, or (0, 0) on
    a device without a reported working-set size. Idempotent: calling
    twice is harmless. Callers must not assume any specific value — read
    the return tuple if they need to know what was applied.
    """
    wired_gb, memory_gb = compute_safe_caps_gb()
    if wired_gb == 0:
        return (0, 0)
    mx.set_wired_limit(wired_gb * 1024**3)
    mx.set_memory_limit(memory_gb * 1024**3)
    return (wired_gb, memory_gb)


__all__ = ["DESIRED_MEMORY_GB", "DESIRED_WIRED_GB", "compute_safe_caps_gb", "install_memory_caps"]
