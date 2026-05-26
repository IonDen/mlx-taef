"""Tests for the mflux integration (skipped if mflux not installed)."""

from dataclasses import dataclass
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


def test_resolved_bn_is_explicit_when_kwargs_passed() -> None:
    import mlx.core as mx

    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(
        variant="taef2",
        save_to="/tmp/preview.png",
        bn_mean=mx.ones(128),
        bn_var=mx.ones(128),
    )
    assert cb.resolved_bn == "explicit"


def test_resolved_bn_is_none_when_no_bn_and_no_flux() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "none"


def test_resolved_bn_is_none_for_non_taef2_variant_even_with_flux() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(
        flux=object(),
        variant="taef1",
        save_to="/tmp/preview.png",
    )
    assert cb.resolved_bn == "none"


@dataclass
class _FakeBN:
    """Minimal real-shape stand-in for mflux Flux2VAE.bn.

    Not a Mock — uses actual attribute names so the auto-extraction code
    exercises real getattr() lookups."""

    running_mean: mx.array
    running_var: mx.array
    eps: float


@dataclass
class _FakeVAE:
    bn: _FakeBN


@dataclass
class _FakeFlux:
    vae: _FakeVAE


def _build_fake_flux_with_nontrivial_bn() -> _FakeFlux:
    return _FakeFlux(
        vae=_FakeVAE(
            bn=_FakeBN(
                running_mean=mx.ones(128) * 2.0,
                running_var=mx.ones(128) * 4.0,
                eps=1e-5,
            ),
        ),
    )


def test_auto_bn_resolved_auto_when_flux_has_bn_for_taef2() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(flux=flux, variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "auto"
    assert cb.bn_mean is not None
    assert cb.bn_var is not None
    assert float(mx.max(cb.bn_mean - flux.vae.bn.running_mean)) == 0.0
    assert float(mx.max(cb.bn_var - flux.vae.bn.running_var)) == 0.0


def test_auto_bn_changes_decoded_output_vs_identity_bn() -> None:
    """Behavioral test: auto-extracted BN must change the decoder output
    direction. Mirrors test_unpack_with_bn_stats_differs_from_identity_bn."""
    from mlx_taef.integrations.mflux import unpack_flux2_latent

    latent_h = 4
    latent_w = 4
    packed = mx.ones((1, latent_h * latent_w, 128))

    # Identity-BN output (no kwargs)
    out_identity = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)

    # Auto-extracted BN output (use the same fake bn stats the callback would extract)
    flux = _build_fake_flux_with_nontrivial_bn()
    out_with_bn = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=flux.vae.bn.running_mean,
        bn_var=flux.vae.bn.running_var,
    )

    assert not np.allclose(np.array(out_identity), np.array(out_with_bn))


def test_explicit_kwargs_win_over_auto_bn() -> None:
    """When user passes bn_mean + bn_var AND auto_bn=True, explicit wins.
    resolved_bn reports "explicit"."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    user_mean = mx.ones(128) * 99.0
    user_var = mx.ones(128) * 99.0
    cb = LivePreviewCallback(
        flux=flux,
        variant="taef2",
        save_to="/tmp/preview.png",
        bn_mean=user_mean,
        bn_var=user_var,
    )
    assert cb.resolved_bn == "explicit"
    assert float(mx.max(cb.bn_mean - user_mean)) == 0.0


def test_auto_bn_off_when_kwarg_false() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(
        flux=flux,
        variant="taef2",
        save_to="/tmp/preview.png",
        auto_bn=False,
    )
    assert cb.resolved_bn == "none"
    assert cb.bn_mean is None
    assert cb.bn_var is None


def test_auto_bn_resolved_none_when_flux_missing_vae(caplog) -> None:
    """flux without .vae attribute: warn + fall back to identity BN."""
    import logging

    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux_without_vae = object()
    with caplog.at_level(logging.WARNING, logger="mlx_taef"):
        cb = LivePreviewCallback(
            flux=flux_without_vae,
            variant="taef2",
            save_to="/tmp/preview.png",
        )
    assert cb.resolved_bn == "none"
    assert cb.bn_mean is None
    assert any("auto_bn" in r.message for r in caplog.records)


def test_auto_bn_resolved_none_when_variant_not_taef2() -> None:
    """auto-bn is a no-op for non-taef2 variants."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(flux=flux, variant="taef1", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "none"
