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

    ok = {"status": "ok", "latent_sha256": "aaaa"}
    failed = {"status": "failed", "latent_sha256": "aaaa"}
    ab._unit_result_path(tmp_path, "lat", "vanilla_vae").write_text(json.dumps(ok))
    ab._unit_result_path(tmp_path, "lat", "bn_inverse").write_text(json.dumps(failed))
    assert ab._pending_units(tmp_path, "lat", latent_sha256="aaaa") == ["bn_inverse", "identity"]


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


def test_pending_units_reruns_results_that_belong_to_a_different_latent(tmp_path: Path) -> None:
    """Catches: a second latent with the same file stem re-scoring the first latent's images."""
    import scripts.ab_taef2_bn_domain as ab

    for condition in ab.CONDITIONS:
        ab._unit_result_path(tmp_path, "lat", condition).write_text(
            json.dumps({"status": "ok", "latent_sha256": "aaaa"})
        )
    assert ab._pending_units(tmp_path, "lat", latent_sha256="aaaa") == []
    assert ab._pending_units(tmp_path, "lat", latent_sha256="bbbb") == list(ab.CONDITIONS)


def test_every_worker_bounds_the_mlx_cache_pool() -> None:
    """Catches: the full-VAE arm (the heaviest) running with MLX's near-device-size default cache
    limit, where retained buffers can sit far above the active-memory watchdog's view."""
    import scripts.ab_taef2_bn_domain as ab

    limits = {c: ab._cache_limit_bytes(c) for c in ab.CONDITIONS}
    assert all(0 < v <= 8 * 1024**3 for v in limits.values())
    assert limits["vanilla_vae"] >= limits["identity"]


def test_score_keeps_a_computed_lpips_when_the_other_arm_fails(tmp_path: Path, monkeypatch) -> None:
    """Catches: one arm's scoring error silently discarding the other arm's valid LPIPS."""
    import scripts.ab_taef2_bn_domain as ab
    import scripts.run_showcase as rs
    from PIL import Image

    rng = np.random.default_rng(0)
    for condition in ab.CONDITIONS:
        pixels = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(ab._unit_image_path(tmp_path, "lat", condition))

    def _fake_scorer(ref: Path, cand: Path) -> float:
        if "identity" in cand.name:
            raise RuntimeError("scorer fell over")
        return 0.25

    monkeypatch.setattr(rs, "_build_lpips_score_fn", lambda: _fake_scorer)

    out = ab._score(tmp_path, "lat", with_lpips=True)

    assert out["scores"]["bn_inverse"]["lpips"] == 0.25
    assert out["scores"]["identity"]["lpips"] is None
    assert "identity" in out["lpips_note"]


def test_pending_units_treats_a_truncated_result_file_as_pending(tmp_path: Path) -> None:
    """Catches: a worker killed mid-write crashing the orchestrator's resume with a JSON error."""
    import scripts.ab_taef2_bn_domain as ab

    ab._unit_result_path(tmp_path, "lat", "vanilla_vae").write_text('{"status": "o')
    assert ab._pending_units(tmp_path, "lat", latent_sha256="aaaa") == list(ab.CONDITIONS)
