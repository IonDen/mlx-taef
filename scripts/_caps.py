"""Shared bench-harness cap constants.

`FULL_VAE_CAP_GB` is used by `scripts/run_showcase.py` and
`scripts/bench_decode.py` workers when running the full mflux VAE as the
baseline condition. Per-condition cap policy: TAEF workers use the
variant `memory_cap_hint_gb` from `mlx_taef.variants`; full-VAE workers
use this constant.
"""

from __future__ import annotations


# Per-flux-variant cap for the full-VAE baseline workers.
# Klein-base-4b full VAE peaks at ~9.6 GB on M1 Max; 12 GB gives headroom.
# Flux.1-dev VAE is smaller (~4-5 GB observed); 6 GB is adequate.
FULL_VAE_CAP_GB: dict[str, int] = {
    "flux1-dev": 6,
    "flux2-klein-base-4b": 12,
}
