"""Decision-logic tests for scripts/ab_taef2_bn_domain.py.

The model-loading workers are exercised by the real run; these tests pin the three places a
silent mistake would corrupt the A/B itself: the domain switch, the resume logic, and the verdict.
"""

import json
from pathlib import Path
from typing import ClassVar

import mlx.core as mx
import numpy as np
import pytest


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
    ab._unit_image_path(tmp_path, "lat", "vanilla_vae").write_bytes(b"png")
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
        ab._unit_image_path(tmp_path, "lat", condition).write_bytes(b"png")
    assert ab._pending_units(tmp_path, "lat", latent_sha256="aaaa") == []
    assert ab._pending_units(tmp_path, "lat", latent_sha256="bbbb") == list(ab.CONDITIONS)


def test_install_worker_limits_maps_each_condition_to_its_wired_cap(monkeypatch) -> None:
    """Catches: the full-VAE arm (the heaviest decode) running under TAEF2's ~1-2 GB wired cap
    because the condition -> bench-condition mapping collapsed to "taef2"."""
    import scripts.ab_taef2_bn_domain as ab
    import scripts.bench_decode as bench

    caps: list[int] = []
    monkeypatch.setattr(bench, "_install_memory_caps", lambda cap: caps.append(cap) or cap)

    assert ab._install_worker_limits("vanilla_vae") == bench._resolve_cap_gb(
        condition="vanilla_vae"
    )
    for condition in ab._TAEF2_CONDITIONS:
        assert ab._install_worker_limits(condition) == bench._resolve_cap_gb(condition="taef2")
    assert caps[0] > caps[1] == caps[2]


def test_every_worker_hands_its_cache_bound_to_the_watchdog(monkeypatch, tmp_path) -> None:
    """Catches: a worker (the full-VAE arm is the heaviest) running with MLX's near-device-size
    default cache limit, or with the watchdog's generic bound instead of the condition's own
    (6 GiB for the full VAE, 2 GiB for the TAEF2 arms). The watchdog installs the bound, so
    the worker must pass it; a second bare `mx.set_cache_limit` call would be overridden."""
    import argparse

    import scripts.ab_taef2_bn_domain as ab
    import scripts.run_showcase as rs

    class _StopError(Exception):
        pass

    class _FakeWatchdog:
        def stop(self) -> None:
            pass

    install_calls: list[dict[str, object]] = []

    def _fake_install(result_path, scenario: str, **kwargs: object) -> _FakeWatchdog:
        install_calls.append(kwargs)
        return _FakeWatchdog()

    monkeypatch.setattr(rs, "_install_live_watchdog", _fake_install)
    monkeypatch.setattr(ab, "_install_worker_limits", lambda condition: 7)
    monkeypatch.setattr(mx, "load", lambda path: (_ for _ in ()).throw(_StopError()))

    for condition in ab.CONDITIONS:
        install_calls.clear()
        args = argparse.Namespace(
            worker=condition, latent=tmp_path / "lat.safetensors", out_dir=tmp_path
        )
        with pytest.raises(_StopError):
            ab._worker_main(args)
        assert install_calls[0]["cache_limit_bytes"] == ab._cache_limit_bytes(condition)
        assert 0 < ab._cache_limit_bytes(condition) <= 8 * 1024**3


def test_unit_result_records_the_watchdog_policy_and_observed_peak(monkeypatch, tmp_path) -> None:
    """Catches: an A/B unit result that states the wired cap it ran under but not the cache
    bound or the observed active+cache peak, so the report cannot say how close it came."""
    import argparse

    import scripts.ab_taef2_bn_domain as ab
    import scripts.run_showcase as rs

    class _FakeWatchdog:
        policy: ClassVar[dict[str, object]] = {"cache_limit_bytes": 2 * 1024**3, "interval_s": 0.05}
        observed: ClassVar[dict[str, object]] = {
            "peak_total_memory_bytes": 5,
            "peak_cache_memory_bytes": 3,
        }

        def stop(self) -> None:
            pass

    monkeypatch.setattr(rs, "_install_live_watchdog", lambda *a, **kw: _FakeWatchdog())
    monkeypatch.setattr(ab, "_install_worker_limits", lambda condition: 7)
    monkeypatch.setattr(
        "scripts.bench_decode._prep_full_vae_flux2",
        lambda latent, h, w: lambda: mx.zeros((1, 2, 2, 3), dtype=mx.uint8),
    )
    monkeypatch.setattr(ab, "_save_png", lambda image, target: target.write_bytes(b"png"))
    latent = tmp_path / "lat.safetensors"
    mx.save_safetensors(
        str(latent), {"latent": mx.zeros((1, 4, 16)), "height": mx.array(16), "width": mx.array(16)}
    )
    args = argparse.Namespace(worker="vanilla_vae", latent=latent, out_dir=tmp_path)

    assert ab._worker_main(args) == 0
    result = json.loads(ab._unit_result_path(tmp_path, "lat", "vanilla_vae").read_text())
    assert result["watchdog"] == _FakeWatchdog.policy
    assert result["watchdog_observed"] == _FakeWatchdog.observed


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


def _fake_worker_factory(fail_on: str | None):
    """A stand-in for the model-loading worker subprocess: writes a tiny PNG + an ok result."""
    import subprocess

    import scripts.ab_taef2_bn_domain as ab
    from PIL import Image

    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        condition = cmd[cmd.index("--worker") + 1]
        latent = Path(cmd[cmd.index("--latent") + 1])
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        if condition == fail_on:
            return subprocess.CompletedProcess(cmd, 70)
        out_dir.mkdir(parents=True, exist_ok=True)
        shade = {"vanilla_vae": 120, "bn_inverse": 60, "identity": 118}[condition]
        Image.new("RGB", (16, 16), (shade, shade, shade)).save(
            ab._unit_image_path(out_dir, latent.stem, condition)
        )
        ab._unit_result_path(out_dir, latent.stem, condition).write_text(
            json.dumps(
                {"status": "ok", "condition": condition, "latent_sha256": ab._sha256(latent)}
            )
        )
        return subprocess.CompletedProcess(cmd, 0)

    return _run


def _orchestrate(tmp_path: Path, monkeypatch, latent_bytes: bytes, fail_on: str | None) -> int:
    import scripts.ab_taef2_bn_domain as ab

    latent = tmp_path / "lat.safetensors"
    latent.write_bytes(latent_bytes)
    monkeypatch.setattr(ab.subprocess, "run", _fake_worker_factory(fail_on))
    monkeypatch.setattr(ab, "_environment", dict)
    return ab.main(["--latent", str(latent), "--out-dir", str(tmp_path / "out"), "--no-lpips"])


def test_a_failed_rerun_on_a_new_latent_leaves_the_published_results_consistent(
    tmp_path: Path, monkeypatch
) -> None:
    """Catches: re-running into a directory that holds another latent's results, failing midway,
    and leaving the old report next to images decoded from the new latent."""
    import scripts.ab_taef2_bn_domain as ab

    out = tmp_path / "out"
    assert _orchestrate(tmp_path, monkeypatch, b"first latent", fail_on=None) == 0
    published = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert "lat.report.json" in published

    assert _orchestrate(tmp_path, monkeypatch, b"second latent", fail_on="identity") == 1

    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == published

    # The retry resumes from the staged units instead of redoing them, then publishes as a whole.
    ran: list[str] = []
    real = _fake_worker_factory(None)
    monkeypatch.setattr(
        ab.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd[cmd.index("--worker") + 1]) or real(cmd),
    )
    latent = tmp_path / "lat.safetensors"
    assert ab.main(["--latent", str(latent), "--out-dir", str(out), "--no-lpips"]) == 0
    assert ran == ["identity"]
    report = json.loads((out / "lat.report.json").read_text())
    assert report["latent_sha256"] == ab._sha256(latent)
    assert {u["latent_sha256"] for u in report["units"].values()} == {ab._sha256(latent)}
    assert [p for p in out.iterdir() if p.is_dir()] == []


def test_a_complete_run_is_not_repeated(tmp_path: Path, monkeypatch) -> None:
    """Catches: the resume check ignoring published results and reloading three models."""
    import scripts.ab_taef2_bn_domain as ab

    assert _orchestrate(tmp_path, monkeypatch, b"same latent", fail_on=None) == 0
    ran: list[str] = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **kw: ran.append("x"))
    latent = tmp_path / "lat.safetensors"
    assert ab.main(["--latent", str(latent), "--out-dir", str(tmp_path / "out"), "--no-lpips"]) == 0
    assert ran == []


def test_the_orchestrator_stops_at_the_first_failed_unit(tmp_path: Path, monkeypatch) -> None:
    """Catches: carrying on after a failed worker, loading every remaining model and assembling
    a report from a run that did not complete."""
    import scripts.ab_taef2_bn_domain as ab

    launched: list[str] = []
    fake = _fake_worker_factory("bn_inverse")

    def _recording(cmd: list[str], **kwargs: object):
        launched.append(cmd[cmd.index("--worker") + 1])
        return fake(cmd)

    latent = tmp_path / "lat.safetensors"
    latent.write_bytes(b"a latent")
    monkeypatch.setattr(ab.subprocess, "run", _recording)
    monkeypatch.setattr(ab, "_environment", dict)
    out = tmp_path / "out"

    assert ab.main(["--latent", str(latent), "--out-dir", str(out), "--no-lpips"]) == 1

    assert launched == ["vanilla_vae", "bn_inverse"]
    assert list(out.glob("*.report.json")) == []
    staged = json.loads(
        ab._unit_result_path(
            ab._staging_dir(out, ab._sha256(latent)), "lat", "bn_inverse"
        ).read_text()
    )
    assert staged["status"] == "failed"


def test_image_stats_report_candidate_minus_reference(tmp_path: Path) -> None:
    """Catches: a flipped sign, which would describe a darker decode as a brighter one."""
    import scripts.ab_taef2_bn_domain as ab
    from PIL import Image

    ref, cand = tmp_path / "ref.png", tmp_path / "cand.png"
    Image.new("RGB", (8, 8), (100, 100, 100)).save(ref)
    Image.new("RGB", (8, 8), (110, 90, 100)).save(cand)

    stats = ab._image_stats(ref, cand)

    assert stats["mean_rgb_shift_255"] == [10.0, -10.0, 0.0]
    assert stats["mae_255"] == pytest.approx(20 / 3, abs=1e-3)


@pytest.mark.parametrize("moves_before_interrupt", [1, 2, 3, 4, 5, 6])
def test_an_interrupted_publish_is_never_mistaken_for_a_complete_result(
    tmp_path: Path, monkeypatch, moves_before_interrupt: int
) -> None:
    """Catches: Ctrl-C between two file moves leaving `--out-dir` looking complete (every later
    run says "already measured") while an image or the report is still stuck in staging."""
    import scripts.ab_taef2_bn_domain as ab

    real_replace = Path.replace
    moved = {"n": 0}

    def _interrupting_replace(self: Path, target: Path) -> Path:
        if moved["n"] >= moves_before_interrupt:
            raise KeyboardInterrupt
        moved["n"] += 1
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _interrupting_replace)
    with pytest.raises(KeyboardInterrupt):
        _orchestrate(tmp_path, monkeypatch, b"a latent", fail_on=None)
    monkeypatch.setattr(Path, "replace", real_replace)

    out, latent = tmp_path / "out", tmp_path / "lat.safetensors"
    assert not ab._is_published(out, "lat", ab._sha256(latent))

    # The next run finishes the job without loading a single model again.
    launched: list[str] = []
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **kw: launched.append("x"))
    assert ab.main(["--latent", str(latent), "--out-dir", str(out), "--no-lpips"]) == 0
    assert launched == []
    assert ab._is_published(out, "lat", ab._sha256(latent))
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"lat.{c}.{ext}" for c in ab.CONDITIONS for ext in ("json", "png")] + ["lat.report.json"]
    )


def test_a_result_with_a_missing_image_is_measured_again(tmp_path: Path, monkeypatch) -> None:
    """Catches: trusting a unit's JSON while the image it describes is gone."""
    import scripts.ab_taef2_bn_domain as ab

    assert _orchestrate(tmp_path, monkeypatch, b"a latent", fail_on=None) == 0
    out, latent = tmp_path / "out", tmp_path / "lat.safetensors"
    ab._unit_image_path(out, "lat", "vanilla_vae").unlink()

    launched: list[str] = []
    fake = _fake_worker_factory(None)
    monkeypatch.setattr(
        ab.subprocess,
        "run",
        lambda cmd, **kw: launched.append(cmd[cmd.index("--worker") + 1]) or fake(cmd),
    )
    assert ab.main(["--latent", str(latent), "--out-dir", str(out), "--no-lpips"]) == 0
    assert launched == ["vanilla_vae"]
    assert ab._unit_image_path(out, "lat", "vanilla_vae").exists()
