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
