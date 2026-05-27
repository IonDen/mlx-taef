import pytest

from mlx_taef.variants import (
    ALL_VARIANTS,
    TAEF1_CONFIG,
    TAEF2_CONFIG,
    TAESD_CONFIG,
    TAESDXL_CONFIG,  # noqa: F401
    TaesdVariantConfig,  # noqa: F401
)


def test_taef2_config_has_correct_latent_channels():
    assert TAEF2_CONFIG.latent_channels == 32


def test_taef2_config_arch_variant_is_flux_2():
    assert TAEF2_CONFIG.arch_variant == "flux_2"
    assert TAEF2_CONFIG.use_midblock_gn is True  # derived from arch_variant


def test_taef2_config_uses_diffusers_key_format():
    assert TAEF2_CONFIG.key_format == "diffusers"
    assert TAEF2_CONFIG.hf_filename == "taef2.safetensors"


def test_taesd_config_uses_upstream_key_format():
    assert TAESD_CONFIG.key_format == "upstream"
    assert TAESD_CONFIG.hf_decoder_filename == "taesd_decoder.safetensors"
    assert TAESD_CONFIG.hf_encoder_filename == "taesd_encoder.safetensors"


def test_taef1_uses_diffusers_single_file():
    assert TAEF1_CONFIG.key_format == "diffusers"
    assert TAEF1_CONFIG.hf_filename == "diffusion_pytorch_model.safetensors"


def test_latent_magnitude_and_shift_match_upstream():
    for v in ALL_VARIANTS:
        assert v.latent_magnitude == 3.0
        assert v.latent_shift == 0.5


def test_variant_config_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        TAEF2_CONFIG.latent_channels = 4  # type: ignore[misc]


def test_all_variants_contains_four_entries():
    assert len(ALL_VARIANTS) == 4
    names = {v.name for v in ALL_VARIANTS}
    assert names == {"taesd", "taesdxl", "taef1", "taef2"}


def test_each_variant_has_memory_cap_hint_gb_field() -> None:
    from mlx_taef.variants import VARIANTS

    for name, cfg in VARIANTS.items():
        assert hasattr(cfg, "memory_cap_hint_gb"), f"{name} missing memory_cap_hint_gb"
        cap = cfg.memory_cap_hint_gb
        assert cap is None or (isinstance(cap, int) and cap > 0), (
            f"{name}.memory_cap_hint_gb must be None or positive int, got {cap!r}"
        )


def test_memory_cap_hint_defaults_per_variant() -> None:
    from mlx_taef.variants import VARIANTS

    assert VARIANTS["taesd"].memory_cap_hint_gb is None
    assert VARIANTS["taesdxl"].memory_cap_hint_gb is None
    assert VARIANTS["taef1"].memory_cap_hint_gb == 1
    assert VARIANTS["taef2"].memory_cap_hint_gb == 2


def test_get_memory_cap_hint_returns_field_value() -> None:
    from mlx_taef.variants import get_memory_cap_hint

    assert get_memory_cap_hint("taef2") == 2
    assert get_memory_cap_hint("taef1") == 1
    assert get_memory_cap_hint("taesd") is None


def test_get_memory_cap_hint_raises_keyerror_on_unknown_variant() -> None:
    import pytest

    from mlx_taef.variants import get_memory_cap_hint

    with pytest.raises(KeyError, match="unknown variant"):
        get_memory_cap_hint("nonexistent")


def test_get_memory_cap_hint_reexported_from_package_root() -> None:
    import mlx_taef

    assert mlx_taef.get_memory_cap_hint("taef2") == 2


def test_conftest_installed_session_wired_cap() -> None:
    """conftest should set MLX wired+memory caps at session start.

    Mirrors mlx-teacache v0.6.0 conftest.py pattern (module-import-time
    cap, not pytest_configure, so the cap lands before any worker module
    is collected). The actual GB value is hardware-dependent — on a 32 GB
    M1 Max it lands at 20 GB; on smaller CI runners (smaller
    max_recommended_working_set_size) it lands lower. The contract is
    that whatever value `_memory_caps.compute_safe_caps_gb` decided is
    what conftest installed.
    """
    import mlx.core as mx

    from mlx_taef._memory_caps import compute_safe_caps_gb
    from tests.conftest import INSTALLED_CAPS_GB

    expected_wired_gb, _ = compute_safe_caps_gb()
    if expected_wired_gb == 0:
        pytest.skip("device does not report max_recommended_working_set_size")

    assert INSTALLED_CAPS_GB[0] == expected_wired_gb, (
        f"conftest installed wired={INSTALLED_CAPS_GB[0]} GB, "
        f"compute_safe_caps_gb returns {expected_wired_gb} GB"
    )

    # mx.set_wired_limit returns the PREVIOUS limit; calling it with the
    # value we expect to be currently installed proves the cap is in place.
    expected_bytes = expected_wired_gb * 1024**3
    previous = mx.set_wired_limit(expected_bytes)
    assert previous == expected_bytes
