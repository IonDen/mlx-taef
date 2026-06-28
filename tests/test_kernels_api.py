from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_taef import TAEF1, TAEF2, TAESD, TAESDXL
from mlx_taef.api import Taef
from mlx_taef.kernels import get_kernel

CONVERTED = Path(__file__).parent / "converted"


def test_classes_bind_their_kernels():
    assert TAESD._kernel is get_kernel("taesd")
    assert TAEF1._kernel is get_kernel("taef1")
    assert TAEF2._kernel is get_kernel("taef2")


def test_channel_flow_decoder_in_dim_matches_latentspec():
    for cls, ch in [(TAESD, 4), (TAESDXL, 4), (TAEF1, 16), (TAEF2, 32)]:
        assert cls().decoder.layers[1].weight.shape[-1] == ch


def test_scale_unscale_roundtrip_reads_kernel_latentspec():
    for cls in (TAESD, TAEF1, TAEF2):
        m = cls()
        raw = mx.clip(mx.random.normal((1, 4, 4, m._kernel.latent.channels)), -2.9, 2.9)
        assert np.allclose(
            np.array(m.unscale_latents(m.scale_latents(raw))), np.array(raw), atol=1e-5
        )


def test_from_kernel_binds_passed_kernel_not_class_default():
    k = get_kernel("taef2")
    m = Taef.from_kernel(k, decoder_path=CONVERTED / "taef2_decoder.safetensors")
    assert m._kernel is k
    assert m.decoder.layers[1].weight.shape[-1] == 32
    assert m.decode(mx.zeros((1, 8, 8, 32))).shape == (1, 64, 64, 3)


def test_zimage_class_binds_zimage_kernel():
    from mlx_taef import ZImage

    assert ZImage._kernel is get_kernel("zimage")
    assert ZImage().decoder.layers[1].weight.shape[-1] == 16
    # `from_kernel` must be the inherited classmethod, not a ZImage override.
    assert ZImage.from_kernel.__func__ is Taef.from_kernel.__func__
