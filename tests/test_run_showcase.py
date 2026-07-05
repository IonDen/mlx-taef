"""Plumbing tests for scripts/run_showcase.py."""

import json
from pathlib import Path

import pytest


def test_argparse_scenario_choices() -> None:
    from scripts.run_showcase import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--scenario", "all"])
    assert args.scenario == "all"

    args = parser.parse_args(["--scenario", "taef2_vs_vae"])
    assert args.scenario == "taef2_vs_vae"

    with pytest.raises(SystemExit):
        parser.parse_args(["--scenario", "nonexistent"])


def test_scenario_dispatch_table() -> None:
    """Each scenario name maps to a worker function."""
    from scripts.run_showcase import _SCENARIO_DISPATCH

    assert "taef2_vs_vae" in _SCENARIO_DISPATCH
    assert "taef1_vs_vae" in _SCENARIO_DISPATCH
    assert "live_preview" in _SCENARIO_DISPATCH
    assert "combined" in _SCENARIO_DISPATCH
    assert "zimage_vs_vae" in _SCENARIO_DISPATCH
    assert "zimage_live_preview" in _SCENARIO_DISPATCH


def test_json_schema_version_round_trip(tmp_path: Path) -> None:
    from scripts.run_showcase import _load_report, _write_report

    report = {
        "schema_version": 1,
        "generated_at": "2026-05-26T00:00:00Z",
        "hardware": {"chip": "Apple M1 Max", "ram_gb": 32},
        "isolation": "subprocess-per-rep",
        "scenarios": {},
    }
    out = tmp_path / "report.json"
    _write_report(out, report)

    loaded = _load_report(out)
    assert loaded == report


def test_json_schema_rejects_unknown_version(tmp_path: Path) -> None:
    from scripts.run_showcase import _load_report

    from mlx_taef.errors import SchemaVersionError

    bad = tmp_path / "report.json"
    bad.write_text(json.dumps({"schema_version": 99, "scenarios": {}}))

    with pytest.raises(SchemaVersionError, match="unknown schema_version"):
        _load_report(bad)


def test_hardware_metadata_includes_imported_versions() -> None:
    """importlib.metadata.version() is used for mlx_taef, mflux. mlx_teacache
    is wrapped in try/except (may be None if not installed)."""
    from scripts.run_showcase import _build_hardware_metadata

    meta = _build_hardware_metadata()
    assert "mlx_taef_version" in meta
    assert "mflux_version" in meta
    assert "mlx_teacache_version" in meta  # may be None if not installed
    # mlx_teacache_version is None or a version string, never a raw exception
    assert meta["mlx_teacache_version"] is None or isinstance(meta["mlx_teacache_version"], str)


def test_compute_ssim_returns_per_pair_array_and_median(tmp_path: Path) -> None:
    """SSIM computed in orchestrator; returned as per-pair list + median."""
    import numpy as np
    from PIL import Image
    from scripts.run_showcase import _compute_ssim

    img_a = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    img_b = img_a.copy()  # identical → SSIM = 1.0
    path_a = tmp_path / "a.webp"
    path_b = tmp_path / "b.webp"
    Image.fromarray(img_a).save(path_a)
    Image.fromarray(img_b).save(path_b)

    result = _compute_ssim([path_a], [path_b])
    assert "ssim_per_pair" in result
    assert "ssim_median" in result
    assert result["ssim_median"] >= 0.99  # near-perfect for identical inputs


def test_sha256_mismatch_warns_but_continues(tmp_path: Path, caplog) -> None:
    import logging

    from scripts.run_showcase import _check_latent_sha

    latent = tmp_path / "latent.safetensors"
    latent.write_bytes(b"actual content")
    sidecar = tmp_path / "latent.safetensors.sha256"
    # Sidecar claims a different hash
    sidecar.write_text(f"{'0' * 64}  latent.safetensors\n")

    with caplog.at_level(logging.WARNING):
        _check_latent_sha(latent)
    assert any("sha mismatch" in r.message.lower() for r in caplog.records)


def test_missing_latent_raises_fixture_latent_missing(tmp_path: Path) -> None:
    from scripts.run_showcase import _check_latent_sha

    from mlx_taef.errors import FixtureLatentMissingError

    missing = tmp_path / "does_not_exist.safetensors"
    with pytest.raises(FixtureLatentMissingError):
        _check_latent_sha(missing)


def test_import_apply_teacache_raises_package_error_when_missing(monkeypatch) -> None:
    import sys

    from mlx_taef.errors import MlxTeacacheNotInstalledError

    monkeypatch.setitem(sys.modules, "mlx_teacache", None)  # forces ImportError
    from scripts.run_showcase import _import_apply_teacache

    with pytest.raises(MlxTeacacheNotInstalledError):
        _import_apply_teacache()


def test_argparse_accepts_zimage_scenarios() -> None:
    from scripts.run_showcase import _build_argparser

    parser = _build_argparser()
    assert parser.parse_args(["--scenario", "zimage_vs_vae"]).scenario == "zimage_vs_vae"
    assert (
        parser.parse_args(["--scenario", "zimage_live_preview"]).scenario == "zimage_live_preview"
    )


def test_run_zimage_vs_vae_uses_correct_condition_and_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked: _run_zimage_vs_vae must call _vs_vae_scenario with the Z-Image condition + fixture."""
    import scripts.run_showcase as rs

    captured: dict[str, object] = {}
    monkeypatch.setattr(rs, "_vs_vae_scenario", lambda **kw: captured.update(kw) or {})
    rs._run_zimage_vs_vae(args=None)
    assert captured["taef_condition"] == "zimage"
    assert captured["flux_variant"] == "z-image-turbo"
    assert captured["latent_name"] == "z_image_turbo.safetensors"


def test_run_scenarios_records_error_and_writes_incrementally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failing scenario is recorded as an error entry, does NOT abort the run, and the
    report is written after each scenario so completed results are never discarded."""
    import argparse

    import scripts.run_showcase as run_showcase

    def _ok(args: object) -> dict[str, str]:
        return {"status": "ok"}

    def _boom(args: object) -> dict[str, str]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_showcase, "_SCENARIO_DISPATCH", {"a": _ok, "b": _boom, "c": _ok})
    writes: list[dict[str, str]] = []

    def _capture_write(path: object, rep: dict[str, object]) -> None:
        # Snapshot per-scenario status at write time — the scenario dicts mutate in place after.
        scenarios = rep["scenarios"]
        assert isinstance(scenarios, dict)
        writes.append({k: v.get("status", "?") for k, v in scenarios.items()})

    monkeypatch.setattr(run_showcase, "_write_report", _capture_write)

    report: dict[str, object] = {"scenarios": {}}
    args = argparse.Namespace(report=tmp_path / "r.json")
    run_showcase._run_scenarios(["a", "b", "c"], args, report)

    assert report["scenarios"]["b"]["error_type"] == "RuntimeError"
    assert report["scenarios"]["c"] == {"status": "ok"}  # ran despite b failing
    # Exact write progression proves the report is written AFTER EACH scenario (an
    # all-then-write-3x implementation would snapshot {a,b,c} three times), and that b's
    # error entry is on disk before c runs:
    assert writes == [
        {"a": "ok"},
        {"a": "ok", "b": "error"},
        {"a": "ok", "b": "error", "c": "ok"},
    ]
