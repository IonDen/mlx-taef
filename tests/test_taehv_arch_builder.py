"""The `taehv` arch builder is registered and returns the right module types."""

import mlx.nn as nn

from mlx_taef.kernels._arch import build_arch
from mlx_taef.kernels._taehv import TaehvDecoder, TaehvEncoder


def test_build_arch_taehv_decoder_and_encoder():
    dec = build_arch("taehv", role="decoder", latent_channels=16, midblock_gn=False)
    enc = build_arch("taehv", role="encoder", latent_channels=16, midblock_gn=False)
    assert isinstance(dec, TaehvDecoder)
    assert isinstance(enc, TaehvEncoder)
    assert isinstance(dec, nn.Sequential)  # subclass — keeps the existing return type
    assert isinstance(enc, nn.Sequential)


def test_build_arch_taehv_ignores_midblock_gn():
    # taehv has no midblock_gn knob; the builder must accept+ignore it (uniform call site).
    a = build_arch("taehv", role="decoder", latent_channels=16, midblock_gn=True)
    b = build_arch("taehv", role="decoder", latent_channels=16, midblock_gn=False)
    assert type(a) is type(b) is TaehvDecoder
