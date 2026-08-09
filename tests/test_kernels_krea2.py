"""Krea 2 kernel: unpack value/rejection + offline equivalence to QwenImage (shared weights)."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.api import Taef
from mlx_taef.kernels import UnpackContext, get_kernel
from mlx_taef.kernels.krea2 import unpack_krea2_latent

CONVERTED = Path(__file__).parent / "converted"


def test_unpack_krea2_latent_full_permutation():
    h, w = 3, 5  # asymmetric: a wrong axis order cannot produce this shape AND these values
    latent = mx.arange(1 * 16 * h * w, dtype=mx.float32).reshape(1, 16, h, w)
    out = unpack_krea2_latent(latent, UnpackContext(latent_height=h, latent_width=w))
    assert out.shape == (1, h, w, 16)
    np.testing.assert_array_equal(np.array(out), np.array(latent).transpose(0, 2, 3, 1))


def test_unpack_krea2_latent_rejects_wrong_channel_axis():
    ctx = UnpackContext(latent_height=4, latent_width=4)
    with pytest.raises(ValueError, match="Krea 2"):
        unpack_krea2_latent(mx.zeros((1, 8, 4, 4)), ctx)  # 8 channels on axis 1, must be 16


def test_unpack_krea2_latent_rejects_wrong_rank():
    ctx = UnpackContext(latent_height=4, latent_width=4)
    with pytest.raises(ValueError, match="Krea 2"):
        unpack_krea2_latent(mx.zeros((16, 4, 4)), ctx)  # 3-D, must be 4-D NCHW


def test_krea2_from_kernel_with_qwen_weights_matches_qwenimage_decode():
    """Reuses QwenImage's committed taew2.1 weights: same arch + 16 channels => bit-identical decode.

    A wrong LatentSpec.channels or ArchSpec on the krea2 kernel makes `from_kernel` build a
    mismatched decoder, so `load_weights(strict=True)` against the taew2.1 file raises (the
    test reds at load, not at the array compare).
    """
    mx.random.seed(0)
    latent = mx.random.normal((1, 16, 16, 16))
    mx.eval(latent)
    weights = CONVERTED / "qwen-image_decoder.safetensors"
    k = Taef.from_kernel(get_kernel("krea2"), decoder_path=weights)
    q = Taef.from_kernel(get_kernel("qwen-image"), decoder_path=weights)
    assert get_kernel("krea2").latent.channels == 16
    out_k = np.array(k.decode(latent))
    out_q = np.array(q.decode(latent))
    assert np.array_equal(out_k, out_q)
    assert out_k.shape == (1, 128, 128, 3)
    assert out_k.min() >= 0.0  # decode() clips to [0,1]
    assert out_k.max() <= 1.0
