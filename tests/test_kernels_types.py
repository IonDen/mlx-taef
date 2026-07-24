import mlx.core as mx
import pytest

from mlx_taef.kernels._types import (
    ArchSpec,
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    UnpackContext,
    WeightSource,
)


def test_latentspec_defaults_and_frozen():
    ls = LatentSpec(channels=16)
    assert (ls.channels, ls.magnitude, ls.shift, ls.downsample) == (16, 3.0, 0.5, 8)
    with pytest.raises((AttributeError, TypeError)):
        ls.channels = 32  # type: ignore[misc]


def test_weightsource_cache_key_always_includes_role():
    diffusers = WeightSource(
        repo="madebyollin/taef1", filename="diffusion_pytorch_model.safetensors"
    )
    assert (
        diffusers.cache_key(role="decoder")
        == "madebyollin_taef1__diffusion_pytorch_model.safetensors__decoder__converter-v1"
    )
    assert (
        diffusers.cache_key(role="encoder")
        == "madebyollin_taef1__diffusion_pytorch_model.safetensors__encoder__converter-v1"
    )
    assert diffusers.cache_key(role="decoder") != diffusers.cache_key(role="encoder")
    upstream = WeightSource(
        repo="madebyollin/taesd",
        decoder_filename="taesd_decoder.safetensors",
        encoder_filename="taesd_encoder.safetensors",
    )
    assert (
        upstream.cache_key(role="decoder")
        == "madebyollin_taesd__taesd_decoder.safetensors__decoder__converter-v1"
    )
    assert (
        upstream.cache_key(role="encoder")
        == "madebyollin_taesd__taesd_encoder.safetensors__encoder__converter-v1"
    )


def test_archspec_is_name_only():
    arch = ArchSpec(name="taesd2d")
    assert arch.name == "taesd2d"
    assert not hasattr(arch, "latent_channels")
    assert not hasattr(arch, "midblock_gn")


def test_unpackcontext_holds_optional_bn_stats_and_eps():
    ctx = UnpackContext(latent_height=32, latent_width=32)
    assert ctx.bn_mean is None
    assert ctx.bn_var is None
    assert ctx.bn_eps == 1e-4


def test_modelkernel_composes_strategies():
    k = ModelKernel(
        name="demo",
        arch=ArchSpec(name="taesd2d"),
        conversion=object(),
        latent=LatentSpec(channels=16),
        source=WeightSource(repo="r", filename="f"),
        integration=MfluxBinding(mflux_models=("demo",), unpack=lambda latent, ctx: latent),
        memory_cap_hint_gb=1,
    )
    assert k.name == "demo"
    assert k.latent.channels == 16
    out = k.integration.unpack(mx.zeros((1,)), UnpackContext(latent_height=1, latent_width=1))
    assert out.shape == (1,)


def test_role_is_exported_with_the_two_roles():
    from typing import get_args

    from mlx_taef.kernels import Role

    assert get_args(Role) == ("decoder", "encoder")


def test_cache_key_tracks_converter_version_constant(monkeypatch) -> None:
    """cache_key must interpolate CONVERTER_VERSION, so bumping it invalidates caches."""
    from mlx_taef.kernels import _types
    from mlx_taef.kernels._types import WeightSource

    source = WeightSource(repo="acme/models", filename="weights.safetensors")
    before = source.cache_key(role="decoder")
    assert f"converter-v{_types.CONVERTER_VERSION}" in before

    monkeypatch.setattr(_types, "CONVERTER_VERSION", 99)
    after = source.cache_key(role="decoder")

    assert "converter-v99" in after
    assert after != before
