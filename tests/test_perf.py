"""Performance and peak-memory budget tests for TAEF2 decode."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_taef import TAEF2

CONVERTED = Path(__file__).parent / "converted" / "taef2_decoder.safetensors"


@pytest.fixture(scope="module")
def taef2_fp16() -> TAEF2:
    """Module-scoped TAEF2 instance at fp16 (decode dtype for perf tests)."""
    return TAEF2.from_pretrained_local(CONVERTED, dtype=mx.float16)


@pytest.mark.benchmark
def test_taef2_decode_1024_under_200ms(taef2_fp16: TAEF2, benchmark) -> None:
    """Decode a 1024x1024 image (128x128 latent) in under 200ms on M-series Mac."""
    latent = mx.random.normal((1, 128, 128, 32)).astype(mx.float16)
    mx.eval(latent)

    def _decode() -> None:
        out = taef2_fp16.decode(latent)
        mx.eval(out)

    _decode()  # warm-up

    benchmark(_decode)
    mean_s = benchmark.stats["mean"]
    assert mean_s < 0.2, f"Decode took {mean_s * 1000:.1f}ms, budget 200ms"


@pytest.mark.benchmark
def test_taef2_decode_peak_memory_under_budget(taef2_fp16: TAEF2) -> None:
    """Peak unified-memory pressure during 1024x1024 decode.

    Budget: 1.5 GB. See COMPARISON.md for the measured TAEF vs full-VAE peak-memory
    numbers (the same-process ~9-10 GB / 5-7x figure was retracted in v0.2.0).
    """
    latent = mx.random.normal((1, 128, 128, 32)).astype(mx.float16)
    mx.eval(latent)
    mx.eval(taef2_fp16.decode(latent))

    mx.reset_peak_memory()
    out = taef2_fp16.decode(latent)
    mx.eval(out)
    peak_bytes = mx.get_peak_memory()
    peak_mb = peak_bytes / 1_000_000

    assert peak_mb < 1500, f"Peak memory {peak_mb:.1f} MB exceeds 1500 MB budget"
