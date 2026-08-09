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
