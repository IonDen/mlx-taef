"""Back-compat shim tests: legacy `mlx_taef.variants` surface derived from kernels."""

import pytest

import mlx_taef.variants as v
from mlx_taef.kernels import get_kernel


def test_legacy_constants_present_and_derived_from_kernels():
    assert v.TAEF2_CONFIG.latent_channels == 32
    assert v.TAEF2_CONFIG.arch_variant == "flux_2"
    assert v.TAEF2_CONFIG.use_midblock_gn is True
    assert v.TAEF2_CONFIG.key_format == "diffusers"
    assert v.TAEF2_CONFIG.hf_repo == get_kernel("taef2").source.repo  # derived, not hardcoded
    assert v.TAESD_CONFIG.key_format == "upstream"
    assert v.TAESD_CONFIG.hf_decoder_filename == "taesd_decoder.safetensors"


def test_non_flux2_variants_have_no_arch_variant():
    # Guards against a MIDBLOCK_GN edit silently flipping a non-taef2 variant.
    for name in ("taesd", "taesdxl", "taef1"):
        cfg = getattr(v, f"{name.upper()}_CONFIG")
        assert cfg.arch_variant is None
        assert cfg.use_midblock_gn is False


def test_all_variants_and_dict_present():
    assert len(v.ALL_VARIANTS) == 4
    assert set(v.VARIANTS) == {"taesd", "taesdxl", "taef1", "taef2"}


def test_get_memory_cap_hint_reexported():
    assert v.get_memory_cap_hint("taef2") == 2
    assert v.get_memory_cap_hint("taesd") is None
    with pytest.raises(KeyError, match="unknown variant"):
        v.get_memory_cap_hint("nope")
