import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.kernels import UnpackContext, get_kernel
from mlx_taef.kernels.flux import unpack_flux1_latent, unpack_flux2_latent


def test_unpack_flux1_rejects_non_64_channels():
    with pytest.raises(ValueError, match="64-channel"):
        unpack_flux1_latent(mx.zeros((1, 16, 16)), UnpackContext(latent_height=4, latent_width=4))


def test_unpack_flux1_matches_mflux_unpack_latents_oracle():
    pytest.importorskip("mflux")
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

    lh = lw = 2
    packed = mx.arange(lh * lw * 64).reshape(1, lh * lw, 64).astype(mx.float32)
    oracle_nchw = FluxLatentCreator.unpack_latents(packed, height=lh * 16, width=lw * 16)
    expected = np.array(mx.transpose(oracle_nchw, (0, 2, 3, 1)))  # NHWC
    out = np.array(unpack_flux1_latent(packed, UnpackContext(latent_height=lh, latent_width=lw)))
    assert out.shape == (1, lh * 2, lw * 2, 16)
    assert np.array_equal(out, expected)


def test_unpack_flux2_value_routing():
    packed = mx.arange(128).reshape(1, 1, 128).astype(mx.float32)
    out = np.array(unpack_flux2_latent(packed, UnpackContext(latent_height=1, latent_width=1)))
    assert out.shape == (1, 2, 2, 32)
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 1, 0] == 1.0
    assert out[0, 1, 0, 0] == 2.0
    assert out[0, 0, 0, 1] == 4.0


def test_binding_dispatch_routes_each_model_to_its_unpack():
    assert get_kernel("taef1").integration.unpack is unpack_flux1_latent
    assert get_kernel("taef2").integration.unpack is unpack_flux2_latent


def test_flux1_callback_end_to_end_writes_preview(monkeypatch, tmp_path):
    pytest.importorskip("mflux")
    from pathlib import Path

    from mlx_taef.api import TAEF1
    from mlx_taef.integrations.mflux import LivePreviewCallback

    converted = Path(__file__).parent / "converted"
    real = TAEF1.from_pretrained_local(converted / "taef1_decoder.safetensors")
    monkeypatch.setattr(TAEF1, "from_pretrained", classmethod(lambda cls, **kw: real))

    lh = lw = 8
    cb = LivePreviewCallback(
        variant="taef1",
        every=1,
        save_to=tmp_path / "p.png",
        latent_height=lh,
        latent_width=lw,
    )
    packed = mx.random.normal((1, lh * lw, 64))
    cb.call_in_loop(t=0, seed=0, prompt="x", latents=packed, config=None, time_steps=None)
    out = tmp_path / "p.png"
    assert out.exists()
    assert out.stat().st_size > 0
