import pytest

from mlx_taef.errors import TaefError, UnknownKernelError
from mlx_taef.kernels import KERNELS, get_kernel


def test_registry_has_exactly_the_five_kernels():
    assert set(KERNELS) == {"taesd", "taesdxl", "taef1", "taef2", "zimage"}


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
        "madebyollin_taef1__diffusion_pytorch_model.safetensors__decoder"
    )


def test_registry_is_immutable():
    with pytest.raises(TypeError):
        KERNELS["x"] = object()  # type: ignore[index]
