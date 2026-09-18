"""Plumbing tests for scripts/_capture_latent.py (mflux.generate_image mocked).

Heavy MLX paths are mocked at the network boundary. Output-path logic and
sha256-sidecar generation run for real against tmp_path.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest


def test_argparse_accepts_known_variants() -> None:
    from scripts._capture_latent import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "flux1-dev", "--out-dir", "/tmp"])
    assert args.variant == "flux1-dev"
    assert args.out_dir == Path("/tmp")


def test_argparse_rejects_unknown_variant() -> None:
    from scripts._capture_latent import _build_argparser

    parser = _build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "nonexistent", "--out-dir", "/tmp"])


def test_sha256_sidecar_is_correct(tmp_path: Path) -> None:
    from scripts._capture_latent import _write_sha256_sidecar

    target = tmp_path / "fake_latent.safetensors"
    target.write_bytes(b"hello world")
    sidecar = _write_sha256_sidecar(target)

    assert sidecar == target.with_suffix(target.suffix + ".sha256")
    content = sidecar.read_text().strip()
    expected_hash = hashlib.sha256(b"hello world").hexdigest()
    assert content.startswith(expected_hash)


def test_main_writes_safetensors_and_sidecar_flux1(tmp_path: Path) -> None:
    """Heavy mflux path mocked; verify the orchestrator writes both files."""
    import mlx.core as mx
    from scripts import _capture_latent

    fake_latent = mx.zeros((1, 16, 32, 32))

    def _fake_capture(**kwargs: object) -> dict[str, mx.array]:
        return {
            "latent": fake_latent,
            "height": mx.array([kwargs["height"]], dtype=mx.int32),  # type: ignore[arg-type]
            "width": mx.array([kwargs["width"]], dtype=mx.int32),  # type: ignore[arg-type]
        }

    with (
        patch.object(_capture_latent, "_capture", side_effect=_fake_capture),
        patch.object(_capture_latent, "_install_memory_caps"),
    ):
        exit_code = _capture_latent.main(
            [
                "--variant",
                "flux1-dev",
                "--out-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    latent_path = tmp_path / "flux1_dev.safetensors"
    sha_path = tmp_path / "flux1_dev.safetensors.sha256"
    assert latent_path.exists()
    assert sha_path.exists()
    # flux1-dev path stores latent + height + width but NOT bn_mean/bn_var.
    saved = mx.load(str(latent_path))
    assert set(saved.keys()) == {"latent", "height", "width"}


def test_main_persists_bn_stats_for_flux2(tmp_path: Path) -> None:
    """flux2-klein-base-4b path also writes bn_mean + bn_var so downstream
    TAEF2 decoders can reproduce the color-correct output without
    re-loading Flux2Klein."""
    import mlx.core as mx
    from scripts import _capture_latent

    fake_latent = mx.zeros((1, 1024, 128))
    fake_bn_mean = mx.zeros((128,))
    fake_bn_var = mx.ones((128,))

    def _fake_capture(**kwargs: object) -> dict[str, mx.array]:
        return {
            "latent": fake_latent,
            "bn_mean": fake_bn_mean,
            "bn_var": fake_bn_var,
            "height": mx.array([kwargs["height"]], dtype=mx.int32),  # type: ignore[arg-type]
            "width": mx.array([kwargs["width"]], dtype=mx.int32),  # type: ignore[arg-type]
        }

    with (
        patch.object(_capture_latent, "_capture", side_effect=_fake_capture),
        patch.object(_capture_latent, "_install_memory_caps"),
    ):
        exit_code = _capture_latent.main(
            [
                "--variant",
                "flux2-klein-base-4b",
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert exit_code == 0
    latent_path = tmp_path / "flux2_klein_base_4b.safetensors"
    assert latent_path.exists()
    saved = mx.load(str(latent_path))
    assert {"latent", "bn_mean", "bn_var", "height", "width"} <= set(saved.keys())
    assert saved["bn_mean"].shape == (128,)
    assert saved["bn_var"].shape == (128,)


def test_capture_flux1_latent_register_generate_retrieve() -> None:
    """Drives the real register→generate→retrieve contract against the REAL mflux
    CallbackRegistry (no model): register() duck-types call_after_loop into after_loop,
    generate dispatches via after_loop_callbacks(), and the captured, eval'd array is returned.
    Using the real registry means an mflux registration/dispatch change reddens this test."""
    import mlx.core as mx
    from mflux.callbacks.callback_registry import CallbackRegistry
    from scripts._capture_latent import _capture_flux1_latent

    class _FiringFlux:
        def __init__(self, latent: mx.array) -> None:
            self._latent = latent
            self.callbacks = CallbackRegistry()

        def generate_image(
            self, *, seed, prompt, num_inference_steps, height, width, guidance
        ) -> None:
            for cb in self.callbacks.after_loop_callbacks():
                cb.call_after_loop(seed=seed, prompt=prompt, latents=self._latent, config=None)

    latent = mx.arange(4, dtype=mx.float32)
    out = _capture_flux1_latent(
        _FiringFlux(latent), prompt="p", seed=0, height=64, width=64, num_steps=1, guidance=1.0
    )
    assert out.shape == (4,)
    assert bool(mx.all(out == latent))


def test_capture_flux1_latent_raises_when_callback_never_fires() -> None:
    """A generation that never dispatches to the callback must raise the package RuntimeError,
    not silently return None. Real CallbackRegistry; generate_image simply never dispatches."""
    from mflux.callbacks.callback_registry import CallbackRegistry
    from scripts._capture_latent import _capture_flux1_latent

    class _SilentFlux:
        def __init__(self) -> None:
            self.callbacks = CallbackRegistry()

        def generate_image(self, **kwargs: object) -> None:
            pass  # never dispatches to after-loop subscribers

    with pytest.raises(RuntimeError, match="did not fire"):
        _capture_flux1_latent(
            _SilentFlux(), prompt="p", seed=0, height=64, width=64, num_steps=1, guidance=1.0
        )


def test_capture_watchdog_abort_skipped_when_generation_already_stopped(
    monkeypatch, tmp_path: Path
) -> None:
    """A breach observed after stop() must not overwrite the capture's real result."""
    import threading

    import scripts._capture_latent as cl

    writes: list[object] = []
    exits: list[int] = []
    monkeypatch.setattr(
        cl.Path, "write_text", lambda self, text: writes.append(text), raising=False
    )
    monkeypatch.setattr(cl.os, "_exit", lambda code: exits.append(code))

    stop_event = threading.Event()
    stop_event.set()
    cl._commit_capture_watchdog_abort(
        tmp_path / "r.abort.json", {"status": "aborted"}, stop_event=stop_event
    )

    assert writes == []
    assert exits == []


def test_capture_watchdog_abort_writes_then_exits_70(monkeypatch, tmp_path: Path) -> None:
    import threading

    import scripts._capture_latent as cl

    events: list[object] = []
    monkeypatch.setattr(
        cl.Path,
        "write_text",
        lambda self, text: events.append(("write", text)),
        raising=False,
    )
    monkeypatch.setattr(cl.os, "_exit", lambda code: events.append(("exit", code)))

    payload = {"status": "aborted", "reason": "memory_ceiling"}
    cl._commit_capture_watchdog_abort(
        tmp_path / "r.abort.json", payload, stop_event=threading.Event()
    )

    assert events[0][0] == "write"
    assert events[1] == ("exit", 70)


def test_capture_watchdog_abort_exits_even_when_write_fails(monkeypatch, tmp_path: Path) -> None:
    """The capture process must still die (honestly, via exit 70) if the abort record
    can't be written — a dead daemon thread with the memory backstop silently gone would
    be worse than a process that exits without an artifact."""
    import threading

    import scripts._capture_latent as cl

    exits: list[int] = []

    def _broken_write(self: object, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cl.Path, "write_text", _broken_write, raising=False)
    monkeypatch.setattr(cl.os, "_exit", lambda code: exits.append(code))

    with pytest.raises(OSError, match="disk full"):
        cl._commit_capture_watchdog_abort(
            tmp_path / "r.abort.json", {"status": "aborted"}, stop_event=threading.Event()
        )
    assert exits == [70]


def test_capture_watchdog_breach_reason_counts_retained_cache_toward_the_ceiling() -> None:
    """Catches: the capture ceiling compared against active memory alone (see the same
    test for scripts/run_showcase.py; the two watchdogs must agree on the accounting)."""
    import scripts._capture_latent as cl

    assert (
        cl._watchdog_breach_reason(
            active_bytes=20, cache_bytes=8, ceiling_bytes=28, elapsed_s=0, wall_budget_s=5
        )
        == "memory_ceiling"
    )
    assert (
        cl._watchdog_breach_reason(
            active_bytes=20, cache_bytes=7, ceiling_bytes=28, elapsed_s=0, wall_budget_s=5
        )
        is None
    )
    assert (
        cl._watchdog_breach_reason(
            active_bytes=20, cache_bytes=7, ceiling_bytes=28, elapsed_s=6, wall_budget_s=5
        )
        == "wall_budget"
    )


def _install_capture_watchdog_with_fake_mlx(
    monkeypatch, tmp_path: Path, *, active_bytes: object, cache_bytes: int
) -> tuple[object, list[dict[str, object]]]:
    """Run the real capture watchdog thread against a fake MLX memory API (32 GiB device).

    `active_bytes` may be an exception instance, raised by the active-memory sample. The
    abort commit is replaced by a recorder that stops the thread, so exactly one payload is
    observed and the process is never exited.
    """
    import threading

    import scripts._capture_latent as cl

    monkeypatch.setattr(cl.mx, "device_info", lambda: {"memory_size": 32 * 1024**3})
    monkeypatch.setattr(cl.mx, "set_cache_limit", lambda n: 0)

    def _active() -> int:
        if isinstance(active_bytes, BaseException):
            raise active_bytes
        return int(active_bytes)  # type: ignore[call-overload]

    monkeypatch.setattr(cl.mx, "get_active_memory", _active)
    monkeypatch.setattr(cl.mx, "get_cache_memory", lambda: cache_bytes)

    payloads: list[dict[str, object]] = []

    def _record_abort(
        abort_path: Path, payload: dict[str, object], *, stop_event: threading.Event
    ) -> None:
        payloads.append(payload)
        stop_event.set()

    monkeypatch.setattr(cl, "_commit_capture_watchdog_abort", _record_abort)
    watchdog = cl._install_capture_watchdog("flux1-dev", tmp_path, interval_s=0.005)
    watchdog._thread.join(timeout=5)
    assert not watchdog._thread.is_alive(), "watchdog thread never reached the abort path"
    return watchdog, payloads


def test_capture_watchdog_aborts_on_active_plus_cache_and_records_both(
    monkeypatch, tmp_path: Path
) -> None:
    """Catches: `_watch` sampling `mx.get_active_memory()` only — 20 GiB active under a
    28 GiB ceiling, but 29 GiB with the retained cache counted."""
    active = 20 * 1024**3
    cache = 9 * 1024**3
    _, payloads = _install_capture_watchdog_with_fake_mlx(
        monkeypatch, tmp_path, active_bytes=active, cache_bytes=cache
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["reason"] == "memory_ceiling"
    assert payload["variant"] == "flux1-dev"
    assert payload["active_memory_bytes"] == active
    assert payload["cache_memory_bytes"] == cache
    assert payload["total_memory_bytes"] == active + cache
    assert payload["ceiling_bytes"] == 28 * 1024**3


def test_capture_watchdog_aborts_explicitly_when_a_memory_sample_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """Catches: a sampling exception killing the daemon thread silently and leaving an
    hour-long capture with no memory backstop."""
    _, payloads = _install_capture_watchdog_with_fake_mlx(
        monkeypatch, tmp_path, active_bytes=RuntimeError("metal device gone"), cache_bytes=0
    )

    assert len(payloads) == 1
    assert payloads[0]["reason"] == "sample_error"
    assert "metal device gone" in str(payloads[0]["error"])
    assert "active_memory_bytes" not in payloads[0]


def test_capture_watchdog_bounds_the_cache_pool_and_records_its_policy(
    monkeypatch, tmp_path: Path
) -> None:
    """Catches: a capture run under MLX's default (near device-size) cache limit, and a
    polling cadence that regressed to the old 0.5 s."""
    import scripts._capture_latent as cl

    limits: list[int] = []
    monkeypatch.setattr(cl.mx, "device_info", lambda: {"memory_size": 32 * 1024**3})
    monkeypatch.setattr(cl.mx, "set_cache_limit", lambda n: limits.append(n) or 0)
    monkeypatch.setattr(cl.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(cl.mx, "get_cache_memory", lambda: 0)

    watchdog = cl._install_capture_watchdog("flux1-dev", tmp_path)
    watchdog.stop()

    assert limits == [cl._CAPTURE_CACHE_LIMIT_BYTES]
    # Captures run the live generation recipes, so they share the live bound's floor.
    from scripts.run_showcase import _LIVE_CACHE_LIMIT_BYTES

    assert cl._CAPTURE_CACHE_LIMIT_BYTES == _LIVE_CACHE_LIMIT_BYTES
    assert watchdog.policy == {
        "ceiling_bytes": 28 * 1024**3,
        "cache_limit_bytes": cl._CAPTURE_CACHE_LIMIT_BYTES,
        "interval_s": 0.05,
        "wall_budget_s": cl._WALL_BUDGET_S,
    }


def test_capture_watchdog_reports_the_observed_active_plus_cache_peak(
    monkeypatch, tmp_path: Path
) -> None:
    """Catches: a capture run that prints its active peak only while the watchdog's own
    reading, active plus cache, came within a few MiB of firing; the high-water marks of the
    sum and of the cache term must both track the samples."""
    import time

    import scripts._capture_latent as cl

    active = iter([1, 5, 2])
    cache = iter([1, 3, 6])
    monkeypatch.setattr(cl.mx, "device_info", lambda: {"memory_size": 32 * 1024**3})
    monkeypatch.setattr(cl.mx, "set_cache_limit", lambda n: 0)
    monkeypatch.setattr(cl.mx, "get_active_memory", lambda: next(active, 2))
    monkeypatch.setattr(cl.mx, "get_cache_memory", lambda: next(cache, 1))

    watchdog = cl._install_capture_watchdog("flux1-dev", tmp_path, interval_s=0.002)
    end = time.monotonic() + 5.0
    while int(watchdog.observed["samples"]) < 3:  # type: ignore[call-overload]
        assert time.monotonic() < end, "watchdog took fewer than 3 samples in 5s"
        time.sleep(0.001)
    watchdog.stop()

    assert watchdog.observed["peak_total_memory_bytes"] == 8
    assert watchdog.observed["peak_cache_memory_bytes"] == 6
    assert watchdog.observed["samples"] >= 3
