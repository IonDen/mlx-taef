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


def test_all_scenario_order_runs_live_scenarios_before_vs_vae() -> None:
    """`--scenario all` must run the live scenarios BEFORE the vs-VAE scenarios.

    The vs-VAE scenarios build LPIPS's torch+AlexNet model (~730 MB resident, never
    returned to the OS) in the orchestrator process. If vs-VAE ran first, the live
    scenarios' subprocess watchdogs (each computing a memory ceiling from the device's
    total memory_size) would run with ~730 MB less real headroom than their math assumes."""
    from scripts.run_showcase import _LIVE_SCENARIOS, _SCENARIO_DISPATCH, _all_scenario_order

    order = _all_scenario_order()

    assert set(order) == set(_SCENARIO_DISPATCH)
    assert len(order) == len(_SCENARIO_DISPATCH)
    live_indices = [i for i, s in enumerate(order) if s in _LIVE_SCENARIOS]
    vs_vae_indices = [i for i, s in enumerate(order) if s not in _LIVE_SCENARIOS]
    assert live_indices, "expected at least one live scenario in the order"
    assert vs_vae_indices, "expected at least one vs-VAE scenario in the order"
    assert max(live_indices) < min(vs_vae_indices)


def test_json_schema_version_round_trip(tmp_path: Path) -> None:
    from scripts.run_showcase import SCHEMA_VERSION, _load_report, _write_report

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-05-26T00:00:00Z",
        "hardware": {"chip": "Apple M1 Max", "ram_gb": 32},
        "isolation": "subprocess-per-condition",
        "scenarios": {},
    }
    out = tmp_path / "report.json"
    _write_report(out, report)

    loaded = _load_report(out)
    assert loaded == report


def test_live_scenario_runs_in_child_and_reads_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    import scripts.run_showcase as rs

    captured: dict[str, object] = {}

    def _fake_run(command: list[str], **kwargs: object) -> object:
        captured["command"] = command
        result_path = Path(command[command.index("--live-result") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text('{"status":"ok","worker":true}')
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(rs.subprocess, "run", _fake_run)
    args = argparse.Namespace(report=tmp_path / "report.json", cap_gb=7)

    result = rs._run_live_scenario_subprocess("live_preview", args)

    assert result == {"status": "ok", "worker": True}
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--live-worker") + 1] == "live_preview"
    assert command[command.index("--cap-gb") + 1] == "7"


def test_live_worker_mode_runs_one_raw_scenario(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_showcase as rs

    calls: list[str] = []
    watchdog_events: list[str] = []

    class _FakeWatchdog:
        def stop(self) -> None:
            watchdog_events.append("stopped")

    def _fake_install(result_path: Path, scenario: str) -> _FakeWatchdog:
        assert result_path == tmp_path / "partial.json"
        watchdog_events.append(f"installed:{scenario}")
        return _FakeWatchdog()

    monkeypatch.setattr(rs, "_install_live_watchdog", _fake_install)
    monkeypatch.setattr(
        rs,
        "_SCENARIO_DISPATCH",
        {"live_preview": lambda args: calls.append("live_preview") or {"status": "ok"}},
    )
    result_path = tmp_path / "partial.json"

    assert (
        rs.main(
            [
                "--live-worker",
                "live_preview",
                "--live-result",
                str(result_path),
                "--no-trash-prior",
            ]
        )
        == 0
    )
    assert calls == ["live_preview"]
    assert watchdog_events == ["installed:live_preview", "stopped"]
    assert json.loads(result_path.read_text()) == {"status": "ok"}


def test_hardware_metadata_names_generation_and_decode_dtypes() -> None:
    from scripts.run_showcase import _build_hardware_metadata

    metadata = _build_hardware_metadata()

    assert metadata["generation_dtype"] == "bf16"
    assert metadata["taef_decode_dtype"] == "float32"
    assert "dtype" not in metadata


def test_hardware_metadata_separates_source_and_installed_versions(monkeypatch) -> None:
    import scripts.run_showcase as rs

    monkeypatch.setattr(rs, "_detect_source_version", lambda: "v0.6.2-3-gabc1234")
    monkeypatch.setattr(
        rs,
        "_pkg_version",
        lambda package: "0.6.2" if package == "mlx-taef" else f"test-{package}",
    )

    metadata = rs._build_hardware_metadata()

    assert metadata["mlx_taef_version"] == "v0.6.2-3-gabc1234"
    assert metadata["mlx_taef_distribution_version"] == "0.6.2"


def test_live_watchdog_breach_reason_checks_memory_before_wall() -> None:
    from scripts.run_showcase import _live_watchdog_breach_reason

    assert (
        _live_watchdog_breach_reason(
            active_bytes=28, ceiling_bytes=28, elapsed_s=10, wall_budget_s=5
        )
        == "memory_ceiling"
    )
    assert (
        _live_watchdog_breach_reason(
            active_bytes=20, ceiling_bytes=28, elapsed_s=6, wall_budget_s=5
        )
        == "wall_budget"
    )
    assert (
        _live_watchdog_breach_reason(
            active_bytes=20, ceiling_bytes=28, elapsed_s=4, wall_budget_s=5
        )
        is None
    )


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
    failures = run_showcase._run_scenarios(["a", "b", "c"], args, report)

    assert failures == 1
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


def test_showcase_main_exits_nonzero_when_a_scenario_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_showcase as rs

    def _boom(args: object) -> dict[str, str]:
        raise RuntimeError("broken scenario")

    monkeypatch.setattr(rs, "_SCENARIO_DISPATCH", {"broken": _boom})

    assert (
        rs.main(
            [
                "--scenario",
                "broken",
                "--report",
                str(tmp_path / "report.json"),
                "--no-trash-prior",
            ]
        )
        == 1
    )


def test_live_generation_uses_strict_callback_and_requires_complete_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    import scripts.run_showcase as rs

    callback_kwargs: dict[str, object] = {}

    class _FakeCallback:
        def __init__(self, **kwargs: object) -> None:
            callback_kwargs.update(kwargs)
            self.save_to = Path(str(kwargs["save_to"]))
            self.saved_paths: list[Path] = []

    class _Registry:
        def __init__(self) -> None:
            self.callback: _FakeCallback | None = None

        def register(self, callback: _FakeCallback) -> None:
            self.callback = callback

    class _FinalImage:
        def save(self, path: Path, *args: object, **kwargs: object) -> None:
            path.write_bytes(b"final")

    class _Flux:
        def __init__(self) -> None:
            self.callbacks = _Registry()

        def generate_image(self, *, num_inference_steps: int, **kwargs: object) -> object:
            callback = self.callbacks.callback
            assert callback is not None
            for idx in range(num_inference_steps):
                path = callback.save_to.with_name(
                    f"{callback.save_to.stem}_step{idx:02d}{callback.save_to.suffix}"
                )
                path.write_bytes(b"frame")
                callback.saved_paths.append(path)
            return type("Generated", (), {"image": _FinalImage()})()

    monkeypatch.setattr(rs, "_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("mlx_taef._memory_caps.install_memory_caps", lambda: (20, 22))
    monkeypatch.setattr("mlx_taef.integrations.mflux.LivePreviewCallback", _FakeCallback)

    result = rs._live_generation(
        args=argparse.Namespace(cap_gb=None),
        model_factory=_Flux,
        callback_variant="taef2",
        latent_height_divisor=16,
        latent_width_divisor=16,
        prompt="test",
        num_steps=2,
        guidance=1.0,
        with_teacache=False,
        auto_bn=True,
        scenario_dir="strict",
    )

    assert callback_kwargs["on_error"] == "raise"
    assert result["preview_count"] == 2
    assert result["status"] == "ok"


def test_live_artifact_validation_rejects_incomplete_gallery(tmp_path: Path) -> None:
    from scripts.run_showcase import _validate_live_artifacts

    from mlx_taef.errors import TaefError

    frame = tmp_path / "step00.webp"
    frame.write_bytes(b"frame")
    final = tmp_path / "final.webp"
    final.write_bytes(b"final")

    with pytest.raises(TaefError, match="expected 2 preview frames, got 1"):
        _validate_live_artifacts([frame], final, expected_count=2)


def test_committed_report_contains_no_absolute_paths() -> None:
    """Path fields in the committed report stay repo-relative — no $HOME or username leak."""
    report_path = Path(__file__).resolve().parent.parent / "_artifacts" / "showcase_report.json"
    report = json.loads(report_path.read_text())

    offenders: list[str] = []

    def _walk(node: object, trail: str, leaf_key: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{trail}.{key}", key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, f"{trail}[{i}]", leaf_key)
        elif (
            isinstance(node, str)
            and node.startswith("/")
            and (
                leaf_key.endswith(("path", "paths", "dir"))
                or leaf_key == "prior_artifacts_moved_to"
            )
        ):
            offenders.append(f"{trail} = {node}")

    _walk(report, "report", "")
    assert offenders == []


def test_watchdog_abort_skipped_when_generation_already_stopped(
    monkeypatch, tmp_path: Path
) -> None:
    """A breach observed after stop() must not overwrite the worker's real result."""
    import threading

    import scripts.run_showcase as rs

    writes: list[object] = []
    exits: list[int] = []
    monkeypatch.setattr(rs, "_write_report", lambda path, payload: writes.append(payload))
    monkeypatch.setattr(rs.os, "_exit", lambda code: exits.append(code))

    stop_event = threading.Event()
    stop_event.set()
    rs._commit_watchdog_abort(tmp_path / "r.json", {"status": "aborted"}, stop_event=stop_event)

    assert writes == []
    assert exits == []


def test_watchdog_abort_writes_then_exits_70(monkeypatch, tmp_path: Path) -> None:
    import threading

    import scripts.run_showcase as rs

    events: list[object] = []
    monkeypatch.setattr(
        rs, "_write_report", lambda path, payload: events.append(("write", payload))
    )
    monkeypatch.setattr(rs.os, "_exit", lambda code: events.append(("exit", code)))

    payload = {"status": "aborted", "reason": "memory_ceiling"}
    rs._commit_watchdog_abort(tmp_path / "r.json", payload, stop_event=threading.Event())

    assert events == [("write", payload), ("exit", 70)]


def test_watchdog_abort_exits_even_when_write_fails(monkeypatch, tmp_path: Path) -> None:
    """The worker must still die (honestly, via exit 70) if the abort record can't be written."""
    import threading

    import scripts.run_showcase as rs

    exits: list[int] = []

    def _broken_write(path: object, payload: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rs, "_write_report", _broken_write)
    monkeypatch.setattr(rs.os, "_exit", lambda code: exits.append(code))

    with pytest.raises(OSError, match="disk full"):
        rs._commit_watchdog_abort(
            tmp_path / "r.json", {"status": "aborted"}, stop_event=threading.Event()
        )
    assert exits == [70]


@pytest.mark.parametrize(
    ("cap_gb", "device_wired_gb", "expected"),
    [
        (5, 20, 5),
        (30, 20, 20),
        (30, 0, None),
    ],
)
def test_resolve_override_wired_gb_clamps_to_device(
    cap_gb: int, device_wired_gb: int, expected: int | None
) -> None:
    """An operator --cap-gb override never exceeds the device ceiling and is skipped
    entirely when the device reports no Metal working-set size."""
    from scripts.run_showcase import _resolve_override_wired_gb

    assert _resolve_override_wired_gb(cap_gb, device_wired_gb) == expected


def test_run_scenarios_routes_live_scenarios_through_subprocess_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live scenarios must go through the watchdog-protected subprocess runner,
    never the in-process dispatch table."""
    import argparse

    import scripts.run_showcase as rs

    routed: list[str] = []
    monkeypatch.setattr(
        rs,
        "_run_live_scenario_subprocess",
        lambda scenario, args: routed.append(scenario) or {"status": "ok"},
    )
    monkeypatch.setattr(
        rs,
        "_SCENARIO_DISPATCH",
        {
            "live_preview": lambda args: pytest.fail(
                "live scenario ran in-process, bypassing the watchdog subprocess"
            )
        },
    )
    args = argparse.Namespace(report=tmp_path / "report.json")
    report: dict[str, object] = {"scenarios": {}}

    failures = rs._run_scenarios(["live_preview"], args, report)

    assert failures == 0
    assert routed == ["live_preview"]
    assert report["scenarios"]["live_preview"] == {"status": "ok"}


def test_vs_vae_worker_installs_active_memory_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vs-VAE per-rep worker (scripts.bench_decode._worker_main) must install the
    same active-memory watchdog the live-scenario worker path uses, not just the
    wired/soft memory caps + subprocess timeout it already had. Mirrors how
    test_live_worker_mode_runs_one_raw_scenario proves the live worker's watchdog
    routing above."""
    import argparse

    import mlx.core as mx
    import scripts.bench_decode as bench

    latent_file = tmp_path / "latent.safetensors"
    mx.save_safetensors(
        str(latent_file),
        {
            "latent": mx.zeros((1, 2, 2, 16)),
            "height": mx.array(16),
            "width": mx.array(16),
        },
    )

    watchdog_events: list[str] = []

    class _FakeWatchdog:
        def stop(self) -> None:
            watchdog_events.append("stopped")

    def _fake_install(result_path: Path, scenario: str, **kwargs: object) -> _FakeWatchdog:
        watchdog_events.append(f"installed:{scenario}")
        return _FakeWatchdog()

    monkeypatch.setattr("scripts.run_showcase._install_live_watchdog", _fake_install)
    monkeypatch.setattr(bench, "_install_memory_caps", lambda cap: 1)
    monkeypatch.setattr(bench, "_prep_taef1", lambda latent, h, w: lambda: "IMG")
    monkeypatch.setattr(bench, "_measure_steady_state", lambda decode_fn: ("IMG", 0.25, 2.0))
    monkeypatch.setattr(bench, "_save_webp", lambda image, target: None)

    args = argparse.Namespace(
        condition="taef1",
        rep=0,
        latent=latent_file,
        save_to=tmp_path / "taef1_rep0.webp",
        applied_cap_gb=1,
        flux_variant="flux1-dev",
    )

    assert bench._worker_main(args) == 0
    assert watchdog_events == ["installed:taef1_rep0", "stopped"]


def test_vs_vae_rep_surfaces_watchdog_abort_reason(tmp_path: Path) -> None:
    """When a rep subprocess exits nonzero AND left a watchdog abort artifact next to
    its save-to path, _run_one_rep must report the abort reason (not just the bare
    exit code) — mirrors how _run_live_scenario_subprocess reads the live worker's
    aborted partial result."""
    import subprocess
    from unittest.mock import patch

    from scripts.bench_decode import _run_one_rep, _watchdog_abort_path

    save_to = tmp_path / "taef1_rep0.webp"
    abort_path = _watchdog_abort_path(save_to)
    abort_path.write_text(
        json.dumps(
            {
                "status": "aborted",
                "scenario": "taef1_rep0",
                "reason": "memory_ceiling",
                "active_memory_bytes": 30 * 1024**3,
                "ceiling_bytes": 28 * 1024**3,
                "elapsed_s": 12.5,
            }
        )
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=70, stdout="", stderr="")

    with patch("scripts.bench_decode.subprocess.run", return_value=fake_proc):
        result = _run_one_rep(
            latent_path=tmp_path / "latent.safetensors",
            condition="taef1",
            flux_variant="flux1-dev",
            rep=0,
            save_to=save_to,
            cap_gb=1,
        )

    assert result["status"] == "failed"
    assert "watchdog aborted: memory_ceiling" in result["error"]


# ---------------------------------------------------------------------------
# LPIPS (offline: injected score_fn only; the real net="alex" scorer is
# network-marked below and needs `uv sync --group fixtures`)
# ---------------------------------------------------------------------------


def test_argparse_no_lpips_flag_defaults_false() -> None:
    from scripts.run_showcase import _build_argparser

    parser = _build_argparser()
    assert parser.parse_args([]).no_lpips is False
    assert parser.parse_args(["--no-lpips"]).no_lpips is True


def test_compute_lpips_returns_per_pair_array_and_median(tmp_path: Path) -> None:
    """A deterministic fake score_fn proves the aggregation shape without lpips/torch."""
    from scripts.run_showcase import _compute_lpips

    path_a = tmp_path / "a.webp"
    path_b = tmp_path / "b.webp"
    path_a.write_bytes(b"fake-a")
    path_b.write_bytes(b"fake-b")

    result = _compute_lpips([path_a], [path_b], score_fn=lambda a, b: 0.25)

    assert result == {"lpips_per_pair": [0.25], "lpips_median": 0.25}


def test_compute_lpips_mismatched_lengths_uses_cross_product(tmp_path: Path) -> None:
    """Mirrors _compute_ssim: every ref against every cand, not a zip."""
    from scripts.run_showcase import _compute_lpips

    refs = [tmp_path / "r0.webp", tmp_path / "r1.webp"]
    cands = [tmp_path / "c0.webp"]
    for p in [*refs, *cands]:
        p.write_bytes(b"x")

    result = _compute_lpips(refs, cands, score_fn=lambda a, b: 0.5)

    assert result["lpips_per_pair"] == [0.5, 0.5]
    assert result["lpips_median"] == 0.5


def test_require_lpips_raises_taef_error_naming_install_and_flag(monkeypatch) -> None:
    """Mirrors test_import_apply_teacache_raises_package_error_when_missing's pattern:
    a None sys.modules entry forces the import to raise ImportError."""
    import sys

    from mlx_taef.errors import TaefError

    monkeypatch.setitem(sys.modules, "lpips", None)
    from scripts.run_showcase import _require_lpips

    with pytest.raises(TaefError) as exc_info:
        _require_lpips()
    message = str(exc_info.value)
    assert "uv sync --group fixtures" in message
    assert "--no-lpips" in message


def test_vs_vae_scenario_no_lpips_flag_skips_lpips_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-lpips must produce explicit empty/None lpips keys, never a silent omission —
    even when both webp lists are non-empty and LPIPS *could* have been computed."""
    import argparse

    import numpy as np
    import scripts.run_showcase as rs
    from PIL import Image

    monkeypatch.setattr(rs, "_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(rs, "_check_latent_sha", lambda latent: {"status": "ok"})

    def _fake_run_orchestrator(
        *,
        latent_path: Path,
        condition: str,
        reps: int,
        save_dir: Path,
        flux_variant: str,
        cap_gb_override: int | None,
    ) -> dict[str, object]:
        save_dir.mkdir(parents=True, exist_ok=True)
        img = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(save_dir / f"{condition}_rep0.webp")
        return {"status": "ok", "median_seconds": 0.1}

    monkeypatch.setattr("scripts.bench_decode._run_orchestrator", _fake_run_orchestrator)

    args = argparse.Namespace(reps=None, cap_gb=None, no_lpips=True)
    result = rs._vs_vae_scenario(
        taef_condition="taef2",
        flux_variant="flux2-klein-base-4b",
        latent_name="flux2_klein_base_4b.safetensors",
        args=args,
    )

    assert result["lpips_per_pair"] == []
    assert result["lpips_median"] is None
    # SSIM was still computed (proves the webps really were discovered — the
    # lpips omission is a deliberate flag effect, not an accidental empty glob).
    assert result["ssim_per_pair"] != []


@pytest.mark.network
def test_compute_lpips_real_scorer_identical_vs_different(tmp_path: Path) -> None:
    """Real LPIPS(net="alex"): identical images -> median ~0.0; clearly different -> > 0.05.

    Network: downloads torchvision AlexNet ImageNet weights on first use. Measured on a
    64x64 synthetic RGB image (seed 0) vs its bitwise inverse: identical median = 0.0,
    different median = 0.098. Weights cache at
    ~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth (sha256
    7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02, ~233 MiB); the
    `lpips` package itself bundles its linear-calibration weights, so only the AlexNet
    backbone is a network download.
    """
    import numpy as np
    from PIL import Image
    from scripts.run_showcase import _compute_lpips

    rng = np.random.default_rng(0)
    img_a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    img_b = img_a.copy()
    img_c = 255 - img_a  # clearly different

    path_a = tmp_path / "a.webp"
    path_b = tmp_path / "b.webp"
    path_c = tmp_path / "c.webp"
    Image.fromarray(img_a).save(path_a)
    Image.fromarray(img_b).save(path_b)
    Image.fromarray(img_c).save(path_c)

    identical = _compute_lpips([path_a], [path_b])
    assert identical["lpips_median"] == pytest.approx(0.0, abs=1e-6)

    different = _compute_lpips([path_a], [path_c])
    assert different["lpips_median"] > 0.05
