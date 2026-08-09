import pytest

from mlx_taef.errors import TaefError, UnknownKernelError
from mlx_taef.kernels import KERNELS, get_kernel


def test_registry_has_exactly_the_seven_kernels():
    assert set(KERNELS) == {
        "taesd",
        "taesdxl",
        "taef1",
        "taef2",
        "zimage",
        "qwen-image",
        "krea2",
    }


def test_qwen_image_kernel_registered():
    k = get_kernel("qwen-image")
    assert k.name == "qwen-image"
    assert k.arch.name == "taehv"
    assert k.latent.channels == 16
    assert k.integration is not None
    assert k.integration.mflux_models == ("qwen-image", "qwen-image-edit")
    assert k.integration.packed_latent_downscale == 16
    assert k.source.sha256 == "04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f"


def test_zimage_shares_taef1_cache_key_and_source():
    z = get_kernel("zimage")
    t = get_kernel("taef1")
    assert z.source.cache_key(role="decoder") == t.source.cache_key(role="decoder")
    assert z.source.cache_key(role="encoder") == t.source.cache_key(role="encoder")
    assert z.source.repo == t.source.repo
    assert z.latent.channels == 16
    assert z.arch is t.arch  # both reference the shared TAESD2D constant from flux.py


def test_kernel_names_byte_identical_and_fixtures_resolve():
    from pathlib import Path

    converted = Path(__file__).parent / "converted"
    for name in ("taesd", "taesdxl", "taef1", "taef2"):
        assert get_kernel(name).name == name
        assert (converted / f"{name}_decoder.safetensors").exists()


def test_get_kernel_unknown_raises_taef_error():
    with pytest.raises(TaefError, match="unknown kernel"):
        get_kernel("nope")
    with pytest.raises(UnknownKernelError):
        get_kernel("nope")


def test_taef1_and_zimage_would_share_cache_key():
    taef1 = get_kernel("taef1")
    assert taef1.source.cache_key(role="decoder") == (
        "madebyollin_taef1__diffusion_pytorch_model.safetensors__decoder__converter-v1"
        "__rev-b1b2d00e9e44__sha-47a6c2bff850"
    )


def test_registry_is_immutable():
    with pytest.raises(TypeError):
        KERNELS["x"] = object()  # type: ignore[index]


def test_midblock_gn_is_kernel_owned_and_compat_view_is_exact() -> None:
    from mlx_taef.kernels import MIDBLOCK_GN

    assert {name: kernel.midblock_gn for name, kernel in KERNELS.items()} == dict(MIDBLOCK_GN)
    assert KERNELS["taef2"].midblock_gn is True
    assert all(kernel.midblock_gn is False for name, kernel in KERNELS.items() if name != "taef2")
