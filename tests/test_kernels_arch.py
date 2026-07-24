from mlx_taef.kernels._arch import ARCH_BUILDERS, build_arch


def test_taesd2d_registered():
    assert "taesd2d" in ARCH_BUILDERS


def test_build_decoder_channels_flow_from_latentspec():
    dec16 = build_arch("taesd2d", role="decoder", latent_channels=16, midblock_gn=False)
    assert dec16.layers[1].weight.shape[-1] == 16
    dec32 = build_arch("taesd2d", role="decoder", latent_channels=32, midblock_gn=True)
    assert dec32.layers[1].weight.shape[-1] == 32


def test_build_encoder_emits_requested_latent_channels():
    enc = build_arch("taesd2d", role="encoder", latent_channels=16, midblock_gn=False)
    assert enc.layers[-1].weight.shape[0] == 16


def test_midblock_gn_toggles_pool_branch():
    with_gn = build_arch("taesd2d", role="decoder", latent_channels=32, midblock_gn=True)
    without = build_arch("taesd2d", role="decoder", latent_channels=32, midblock_gn=False)
    assert with_gn.layers[3].pool is not None
    assert without.layers[3].pool is None


def test_unknown_arch_raises():
    import pytest

    from mlx_taef.errors import UnknownArchitectureError

    with pytest.raises(UnknownArchitectureError, match="unknown arch"):
        build_arch("nope", role="decoder", latent_channels=16, midblock_gn=False)


def test_unknown_role_raises_package_architecture_error() -> None:
    import pytest

    from mlx_taef.errors import UnknownArchitectureError

    with pytest.raises(UnknownArchitectureError, match="unknown arch/role"):
        build_arch("taesd2d", role="sideways", latent_channels=16, midblock_gn=False)
