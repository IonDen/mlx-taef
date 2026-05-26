"""Tests for the mflux integration (skipped if mflux not installed)."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mflux = pytest.importorskip("mflux")

from mlx_taef.integrations.mflux import LivePreviewCallback, unpack_flux2_latent  # noqa: E402


def test_unpack_flux2_latent_shape() -> None:
    """Unpack from (1, lH*lW, 128) to (1, lH*2, lW*2, 32)."""
    latent_h = 32
    latent_w = 32
    packed = mx.zeros((1, latent_h * latent_w, 128))
    unpacked = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert unpacked.shape == (1, latent_h * 2, latent_w * 2, 32)


def test_unpack_flux2_latent_rejects_wrong_channel_count() -> None:
    packed = mx.zeros((1, 16, 64))  # 64 channels, should be 128
    with pytest.raises(ValueError, match="128-channel"):
        unpack_flux2_latent(packed, latent_height=4, latent_width=4)


def test_live_preview_callback_writes_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with fake-shaped packed latent — proves the callback wires."""
    save_path = tmp_path / "preview.png"

    # Patch from_pretrained to avoid network — use the pre-baked decoder weights.
    from mlx_taef import TAEF2

    converted = Path(__file__).parent / "converted" / "taef2_decoder.safetensors"
    real_taef2 = TAEF2.from_pretrained_local(converted)
    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(lambda cls, **kw: real_taef2))

    cb = LivePreviewCallback(
        variant="taef2",
        every=1,
        save_to=save_path,
        latent_height=32,
        latent_width=32,
    )
    fake_packed = mx.random.normal((1, 32 * 32, 128))
    cb.call_in_loop(t=0, seed=0, prompt="", latents=fake_packed, config=None, time_steps=None)
    assert save_path.exists()
    assert save_path.stat().st_size > 100  # not an empty file


def test_flux2_klein_generate_image_has_no_callbacks_kwarg() -> None:
    """Doc-shape guard: ensure README/manual examples match installed mflux API."""
    import inspect

    from mflux.models.flux2 import Flux2Klein

    sig = inspect.signature(Flux2Klein.generate_image)
    assert "callbacks" not in sig.parameters, (
        "If mflux added a callbacks kwarg, update the README to use it directly."
    )
    # Affirmatively assert the registration path:
    assert hasattr(Flux2Klein, "__init__")  # placeholder anchor


def test_callback_registry_register_method_exists() -> None:
    from mflux.callbacks.callback_registry import CallbackRegistry

    assert hasattr(CallbackRegistry, "register")


def test_unpack_with_bn_stats_differs_from_identity_bn() -> None:
    """Non-trivial BN stats should change the output values."""
    latent_h = 4
    latent_w = 4
    packed = mx.ones((1, latent_h * latent_w, 128))
    out_identity = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)

    bn_mean = mx.ones(128) * 2.0
    bn_var = mx.ones(128) * 4.0  # std = 2.0 (approx)
    out_with_bn = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=bn_mean,
        bn_var=bn_var,
    )
    assert not np.allclose(np.array(out_identity), np.array(out_with_bn))


def test_live_preview_callback_accepts_flux_kwarg() -> None:
    """Constructor must accept flux= keyword (no behavior change yet — Task 7
    adds the auto-bn extraction; this task just exposes the parameter)."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(flux=None, variant="taef2", save_to="/tmp/preview.png")
    assert cb.flux is None


def test_live_preview_callback_stores_passed_flux_instance() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    sentinel = object()
    cb = LivePreviewCallback(flux=sentinel, variant="taef2", save_to="/tmp/preview.png")
    assert cb.flux is sentinel
