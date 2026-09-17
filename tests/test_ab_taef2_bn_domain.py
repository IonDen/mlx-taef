"""Decision-logic tests for scripts/ab_taef2_bn_domain.py.

The model-loading workers are exercised by the real run; these tests pin the three places a
silent mistake would corrupt the A/B itself: the domain switch, the resume logic, and the verdict.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def _packed_and_stats() -> tuple[mx.array, mx.array, mx.array]:
    lh, lw = 2, 3
    packed = mx.arange(lh * lw * 128).reshape(1, lh * lw, 128).astype(mx.float32) / 100.0
    bn_mean = mx.arange(128).astype(mx.float32) / 64.0
    bn_var = mx.full((128,), 3.0)
    return packed, bn_mean, bn_var


def test_identity_condition_ignores_the_bn_stats_and_inverse_applies_them() -> None:
    """Catches: both conditions decoding the same domain (an A/B that compares A with A)."""
    import scripts.ab_taef2_bn_domain as ab

    from mlx_taef.integrations.mflux import unpack_flux2_latent

    packed, bn_mean, bn_var = _packed_and_stats()
    kw = {"latent_height": 2, "latent_width": 3}
    identity = np.array(ab._unpack_for("identity", packed, 32, 48, bn_mean, bn_var))
    inverse = np.array(ab._unpack_for("bn_inverse", packed, 32, 48, bn_mean, bn_var))

    assert np.array_equal(identity, np.array(unpack_flux2_latent(packed, **kw)))
    assert np.array_equal(
        inverse, np.array(unpack_flux2_latent(packed, bn_mean=bn_mean, bn_var=bn_var, **kw))
    )
    assert not np.allclose(identity, inverse)


def test_pending_units_skips_ok_results_and_retries_failed_or_missing(tmp_path: Path) -> None:
    """Catches: a resume that re-runs finished units, or trusts a failed unit's file."""
    import scripts.ab_taef2_bn_domain as ab

    ab._unit_result_path(tmp_path, "lat", "vanilla_vae").write_text(json.dumps({"status": "ok"}))
    ab._unit_result_path(tmp_path, "lat", "bn_inverse").write_text(json.dumps({"status": "failed"}))
    assert ab._pending_units(tmp_path, "lat") == ["bn_inverse", "identity"]


def test_verdict_needs_both_metrics_to_agree_and_treats_lpips_as_lower_is_better() -> None:
    """Catches: reading LPIPS as higher-is-better, or crowning a winner on a split decision."""
    import scripts.ab_taef2_bn_domain as ab

    clear = ab._verdict(
        {"bn_inverse": {"ssim": 0.62, "lpips": 0.40}, "identity": {"ssim": 0.90, "lpips": 0.10}}
    )
    assert clear["winner"] == "identity"

    split = ab._verdict(
        {"bn_inverse": {"ssim": 0.62, "lpips": 0.10}, "identity": {"ssim": 0.90, "lpips": 0.40}}
    )
    assert split["winner"] == "inconclusive"

    close = ab._verdict(
        {"bn_inverse": {"ssim": 0.700, "lpips": 0.30}, "identity": {"ssim": 0.705, "lpips": 0.29}}
    )
    assert close["winner"] == "inconclusive"


def test_verdict_without_lpips_decides_on_ssim_margin_alone() -> None:
    """Catches: a missing LPIPS scorer (no torch) silently blocking any verdict."""
    import scripts.ab_taef2_bn_domain as ab

    out = ab._verdict(
        {"bn_inverse": {"ssim": 0.85, "lpips": None}, "identity": {"ssim": 0.60, "lpips": None}}
    )
    assert out["winner"] == "bn_inverse"
