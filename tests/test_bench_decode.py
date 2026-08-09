"""Plumbing tests for scripts/bench_decode.py.

MLX-heavy paths mocked at the network/model-load boundary. Sentinel
parsing, JSON schema, dispatch table run for real.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_steady_state_measurement_warms_before_reset_and_timer(monkeypatch) -> None:
    import mlx.core as mx
    import scripts.bench_decode as bench

    events: list[str] = []
    calls = 0

    def _decode() -> str:
        nonlocal calls
        calls += 1
        events.append(f"decode-{calls}")
        return f"image-{calls}"

    monkeypatch.setattr(mx, "eval", lambda image: events.append(f"eval-{image}"))
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: events.append("reset"))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 2 * 1024**3)
    times = iter([10.0, 10.25])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))

    image, elapsed_s, peak_gb = bench._measure_steady_state(_decode)

    assert events == ["decode-1", "eval-image-1", "reset", "decode-2", "eval-image-2"]
    assert image == "image-2"
    assert elapsed_s == 0.25
    assert peak_gb == 2.0


def test_argparse_orchestrator_mode() -> None:
    from scripts.bench_decode import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(
        [
            "--latent",
            "/tmp/x.safetensors",
            "--condition",
            "taef2",
            "--reps",
            "5",
        ]
    )
    assert args.latent == Path("/tmp/x.safetensors")
    assert args.condition == "taef2"
    assert args.reps == 5
    assert not args.worker_mode


def test_argparse_worker_mode() -> None:
    from scripts.bench_decode import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(
        [
            "--worker-mode",
            "--latent",
            "/tmp/x.safetensors",
            "--condition",
            "taef2",
            "--rep",
            "0",
            "--save-to",
            "/tmp/out.webp",
            "--applied-cap-gb",
            "2",
        ]
    )
    assert args.worker_mode
    assert args.rep == 0
    assert args.save_to == Path("/tmp/out.webp")
    assert args.applied_cap_gb == 2


def test_sentinel_emission_is_first_of_line() -> None:
    from scripts.bench_decode import _emit_sentinel

    result = {
        "condition": "taef2",
        "rep": 0,
        "elapsed_s": 0.1,
        "peak_memory_gb": 1.5,
        "applied_cap_gb": 2,
        "image_path": "/tmp/out.webp",
    }
    line = _emit_sentinel(result)
    assert line.startswith("::BENCH_RESULT::")
    assert "\n" not in line  # one-liner
    payload = json.loads(line[len("::BENCH_RESULT::") :])
    assert payload == result


def test_parse_sentinel_extracts_first_of_line(tmp_path: Path) -> None:
    from scripts.bench_decode import _parse_worker_stdout

    stdout = "\n".join(
        [
            "Loading model...",
            "Decode: 0.094s",
            "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0, "elapsed_s": 0.094}),
            "Done.",
        ]
    )
    parsed = _parse_worker_stdout(stdout)
    assert parsed["condition"] == "taef2"
    assert parsed["rep"] == 0


def test_parse_sentinel_ignores_mid_line_occurrences() -> None:
    """The sentinel string can appear inside a debug print without being
    a sentinel — parser must require line-start."""
    from scripts.bench_decode import _parse_worker_stdout

    stdout = "\n".join(
        [
            "Debug log mentioning ::BENCH_RESULT:: in passing",
            "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0, "elapsed_s": 0.094}),
        ]
    )
    parsed = _parse_worker_stdout(stdout)
    assert parsed["rep"] == 0  # picked the line-start one


def test_parse_sentinel_raises_on_multiple_sentinels() -> None:
    """Deliberate divergence from mlx-teacache: exactly one sentinel per
    worker. Multiple is a contract violation."""
    from scripts.bench_decode import _parse_worker_stdout

    from mlx_taef.errors import TaefError

    stdout = "\n".join(
        [
            "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0}),
            "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0}),
        ]
    )
    with pytest.raises(TaefError, match="multiple sentinels"):
        _parse_worker_stdout(stdout)


def test_parse_sentinel_raises_on_missing_sentinel() -> None:
    from scripts.bench_decode import _parse_worker_stdout

    from mlx_taef.errors import TaefError

    stdout = "Just some output, no sentinel."
    with pytest.raises(TaefError, match="no sentinel"):
        _parse_worker_stdout(stdout)


def test_orchestrator_dispatch_per_condition_cap_split() -> None:
    """TAEF worker gets variant memory cap; full-VAE worker gets FULL_VAE_CAP_GB."""
    from scripts.bench_decode import _resolve_cap_gb

    assert _resolve_cap_gb(condition="taef2") == 2
    assert _resolve_cap_gb(condition="taef1") == 1
    assert _resolve_cap_gb(condition="vanilla_vae", flux_variant="flux1-dev") == 6
    assert _resolve_cap_gb(condition="vanilla_vae", flux_variant="flux2-klein-base-4b") == 12


def test_resolve_cap_gb_raises_on_unknown_condition() -> None:
    """The success paths are covered above; the unknown-condition guard at the
    bottom of _resolve_cap_gb must raise TaefError, not fall through / return None."""
    from scripts.bench_decode import _resolve_cap_gb

    from mlx_taef.errors import TaefError

    with pytest.raises(TaefError, match="unknown condition"):
        _resolve_cap_gb(condition="totally-bogus")


def test_orchestrator_rejects_partial_rep_sample_after_attempting_all_reps() -> None:
    """A release median must never be computed from an undersized survivor sample."""
    from scripts.bench_decode import _run_orchestrator

    from mlx_taef.errors import TaefError

    with patch("scripts.bench_decode._run_one_rep") as mock_rep:
        # Rep 0 fails, rep 1 succeeds, rep 2 succeeds.
        mock_rep.side_effect = [
            {"condition": "taef2", "rep": 0, "status": "failed", "error": "OOM"},
            {"condition": "taef2", "rep": 1, "elapsed_s": 0.1, "peak_memory_gb": 1.5},
            {"condition": "taef2", "rep": 2, "elapsed_s": 0.1, "peak_memory_gb": 1.5},
        ]
        with pytest.raises(TaefError, match=r"1 of 3 reps failed.*OOM"):
            _run_orchestrator(
                latent_path=Path("/tmp/x.safetensors"),
                condition="taef2",
                reps=3,
                save_dir=Path("/tmp"),
            )

    assert mock_rep.call_count == 3


def test_orchestrator_exits_nonzero_if_all_reps_fail() -> None:
    """All reps failed → orchestrator must surface the failure."""
    from scripts.bench_decode import _run_orchestrator

    from mlx_taef.errors import TaefError

    with patch("scripts.bench_decode._run_one_rep") as mock_rep:
        mock_rep.side_effect = [
            {"condition": "taef2", "rep": i, "status": "failed", "error": "boom"} for i in range(3)
        ]
        with pytest.raises(TaefError, match="all reps failed"):
            _run_orchestrator(
                latent_path=Path("/tmp/x.safetensors"),
                condition="taef2",
                reps=3,
                save_dir=Path("/tmp"),
            )


def test_cap_gb_override_threads_through_to_worker() -> None:
    """`--cap-gb` on run_showcase must reach the worker via the orchestrator,
    not get silently dropped (codex audit finding #5)."""
    from scripts.bench_decode import _run_orchestrator

    with patch("scripts.bench_decode._run_one_rep") as mock_rep:
        mock_rep.return_value = {
            "condition": "taef2",
            "rep": 0,
            "elapsed_s": 0.1,
            "peak_memory_gb": 1.5,
        }
        result = _run_orchestrator(
            latent_path=Path("/tmp/x.safetensors"),
            condition="taef2",
            reps=1,
            save_dir=Path("/tmp"),
            cap_gb_override=4,
        )
    assert result["applied_cap_gb"] == 4
    # Verify _run_one_rep was called with the override, not the default (taef2=2)
    _, kwargs = mock_rep.call_args
    assert kwargs["cap_gb"] == 4


def test_resolve_cap_gb_zimage_uses_registry_hint() -> None:
    from scripts.bench_decode import _resolve_cap_gb

    assert _resolve_cap_gb(condition="zimage") == 1  # KERNELS["zimage"].memory_cap_hint_gb


def test_decode_zimage_returns_uint8_nhwc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: the zimage decode thunk must return decode_image() directly (uint8
    NHWC), NOT pass it through _decoded_to_uint8_nhwc (which would transpose/renormalize to
    garbage). _prep_zimage builds the thunk; calling it runs the decode."""
    from pathlib import Path

    import mlx.core as mx
    from scripts.bench_decode import _prep_zimage

    from mlx_taef import ZImage

    weights = Path("tests/converted/taef1_decoder.safetensors")
    real = ZImage.from_pretrained_local(weights)
    monkeypatch.setattr(ZImage, "from_pretrained", classmethod(lambda cls, **kw: real))
    decode_fn = _prep_zimage(mx.zeros((16, 1, 8, 8)), 64, 64)
    out = decode_fn()
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == mx.uint8


def test_malformed_sentinel_json_marks_rep_failed_not_aborted() -> None:
    """Codex audit / subprocess reviewer finding #2: a json.JSONDecodeError
    in the worker's sentinel must NOT abort the entire orchestrator —
    it should be re-raised as TaefError and turned into a failed-rep
    record by _run_one_rep."""
    import subprocess

    from scripts.bench_decode import _parse_worker_stdout

    from mlx_taef.errors import TaefError

    # Direct test of _parse_worker_stdout
    stdout = "some logs\n::BENCH_RESULT::{not json at all\nbye\n"
    with pytest.raises(TaefError, match="malformed sentinel JSON"):
        _parse_worker_stdout(stdout)

    # And via _run_one_rep — make sure it's converted to a failed dict
    from scripts.bench_decode import _run_one_rep

    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    with patch("scripts.bench_decode.subprocess.run", return_value=fake_proc):
        result = _run_one_rep(
            latent_path=Path("/tmp/x"),
            condition="taef2",
            flux_variant="flux2-klein-base-4b",
            rep=0,
            save_to=Path("/tmp/out.webp"),
            cap_gb=2,
        )
    assert result["status"] == "failed"
    assert "malformed sentinel JSON" in result["error"]


def test_repo_relative_maps_artifact_paths_and_foreign_paths(tmp_path: Path) -> None:
    from scripts.bench_decode import _REPO_ROOT, _repo_relative

    inside = _REPO_ROOT / "_artifacts" / "showcase" / "taef2" / "taef" / "taef2_rep0.webp"
    assert _repo_relative(inside) == "_artifacts/showcase/taef2/taef/taef2_rep0.webp"

    outside = tmp_path / "elsewhere" / "out.webp"
    assert _repo_relative(outside) == "out.webp"


def test_worker_main_clears_stale_watchdog_abort_artifact_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a rep that crashes for a REAL reason must not be misreported by
    _run_one_rep as a watchdog abort just because a PRIOR rep's stale abort artifact
    still exists at the same save-to path (save-to paths are reused across bench runs).
    _worker_main must clear any pre-existing artifact before doing any real work, so
    presence of the file after a nonzero exit always means THIS rep's watchdog fired."""
    import argparse

    import mlx.core as mx
    import scripts.bench_decode as bench
    import scripts.run_showcase as run_showcase

    latent_file = tmp_path / "latent.safetensors"
    mx.save_safetensors(
        str(latent_file),
        {
            "latent": mx.zeros((1, 2, 2, 16)),
            "height": mx.array(16),
            "width": mx.array(16),
        },
    )
    save_to = tmp_path / "out.webp"
    abort_path = bench._watchdog_abort_path(save_to)
    abort_path.parent.mkdir(parents=True, exist_ok=True)
    abort_path.write_text(
        json.dumps({"status": "aborted", "reason": "memory_ceiling", "rep": "stale-prior-rep"})
    )

    class _FakeWatchdog:
        def stop(self) -> None:
            pass

    monkeypatch.setattr(bench, "_install_memory_caps", lambda cap: 1)
    monkeypatch.setattr(run_showcase, "_install_live_watchdog", lambda *a, **kw: _FakeWatchdog())

    def _boom(latent: object, h: object, w: object) -> None:
        raise ValueError("boom: real crash, not a watchdog abort")

    monkeypatch.setattr(bench, "_prep_taef1", _boom)

    args = argparse.Namespace(
        condition="taef1",
        rep=0,
        latent=latent_file,
        save_to=save_to,
        applied_cap_gb=1,
        flux_variant="flux1-dev",
    )

    with pytest.raises(ValueError, match="boom: real crash"):
        bench._worker_main(args)

    assert not abort_path.exists()


def test_worker_main_routes_through_steady_state_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_worker_main must time via _measure_steady_state (warmup included), not inline."""
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

    def _fake_decode() -> str:
        return "IMG"

    class _FakeWatchdog:
        def stop(self) -> None:
            pass

    measured: list[object] = []
    monkeypatch.setattr(bench, "_install_memory_caps", lambda cap: 1)
    # Offline test: must not spin up a real _install_live_watchdog thread (which reads real
    # mx.device_info() and starts a live daemon thread) — mirrors
    # test_vs_vae_worker_installs_active_memory_watchdog in tests/test_run_showcase.py.
    monkeypatch.setattr(
        "scripts.run_showcase._install_live_watchdog", lambda *a, **kw: _FakeWatchdog()
    )
    monkeypatch.setattr(bench, "_prep_taef1", lambda latent, h, w: _fake_decode)
    monkeypatch.setattr(
        bench,
        "_measure_steady_state",
        lambda decode_fn: measured.append(decode_fn) or ("IMG", 0.25, 2.0),
    )
    monkeypatch.setattr(bench, "_save_webp", lambda image, target: None)

    args = argparse.Namespace(
        condition="taef1",
        rep=0,
        latent=latent_file,
        save_to=tmp_path / "out.webp",
        applied_cap_gb=1,
        flux_variant="flux1-dev",
    )

    assert bench._worker_main(args) == 0
    assert measured == [_fake_decode]
    assert '"elapsed_s": 0.25' in capsys.readouterr().out


def test_worker_wall_budget_stays_strictly_under_subprocess_timeout() -> None:
    """The watchdog's wall arm is a backstop under the subprocess timeout — if the two
    budgets are equal or the watchdog's is longer, the subprocess.run(timeout=...) always
    fires first and `reason="wall_budget"` is unreachable in practice."""
    from scripts.bench_decode import _SUBPROCESS_TIMEOUT_S, _worker_wall_budget_s

    for condition, timeout_s in _SUBPROCESS_TIMEOUT_S.items():
        budget_s = _worker_wall_budget_s(condition)
        assert budget_s < timeout_s


def test_worker_main_installs_watchdog_with_condition_scoped_wall_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_worker_main must pass an explicit wall_budget_s under this condition's subprocess
    timeout, not the run_showcase default (3300s) which sits far above every
    _SUBPROCESS_TIMEOUT_S entry and makes the wall arm dead."""
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

    class _FakeWatchdog:
        def stop(self) -> None:
            pass

    install_calls: list[dict[str, object]] = []

    def _fake_install(result_path: Path, scenario: str, **kwargs: object) -> _FakeWatchdog:
        install_calls.append(kwargs)
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
        save_to=tmp_path / "out.webp",
        applied_cap_gb=1,
        flux_variant="flux1-dev",
    )

    assert bench._worker_main(args) == 0
    assert install_calls == [{"wall_budget_s": bench._worker_wall_budget_s("taef1")}]
