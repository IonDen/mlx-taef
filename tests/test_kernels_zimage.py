"""Z-Image kernel: unpack value/rejection + offline equivalence to TAEF1."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.api import Taef
from mlx_taef.kernels import UnpackContext, get_kernel
from mlx_taef.kernels.zimage import unpack_zimage_latent

CONVERTED = Path(__file__).parent / "converted"


def test_unpack_zimage_latent_value_and_shape():
    """Asymmetric (16,1,2,3) fixture: input [c,0,h,w] must land at output [0,h,w,c]."""
    base = np.zeros((16, 1, 2, 3), dtype=np.float32)
    for c in range(16):
        for h in range(2):
            for w in range(3):
                base[c, 0, h, w] = c * 100 + h * 10 + w
    latent = mx.array(base)
    ctx = UnpackContext(latent_height=2, latent_width=3)
    out = unpack_zimage_latent(latent, ctx)
    assert out.shape == (1, 2, 3, 16)
    assert float(out[0, 1, 2, 5]) == 512.0  # input [5,0,1,2]
    assert float(out[0, 0, 1, 3]) == 301.0  # input [3,0,0,1]


def test_unpack_zimage_latent_rejects_wrong_axis_layout():
    """A 4D latent whose channel axis isn't 16 at axis 0 must raise (e.g. an already-batched one)."""
    ctx = UnpackContext(latent_height=4, latent_width=4)
    with pytest.raises(ValueError, match=r"16, 1, h, w"):
        unpack_zimage_latent(mx.zeros((1, 16, 4, 4)), ctx)  # shape[0]==1 != 16


def test_unpack_zimage_latent_rejects_wrong_channel_count():
    ctx = UnpackContext(latent_height=4, latent_width=4)
    with pytest.raises(ValueError, match=r"16, 1, h, w"):
        unpack_zimage_latent(mx.zeros((32, 1, 4, 4)), ctx)  # 32 channels, must be 16


def test_zimage_from_kernel_with_taef1_weights_matches_taef1_decode():
    """Reuses TAEF1's committed weights: same arch + 16 channels => bit-identical decode.

    The catch: a wrong LatentSpec.channels or ArchSpec on the zimage kernel makes
    `from_kernel` build a mismatched decoder, so `load_weights(strict=True)` against the
    16-ch taef1 file raises (the test reds at load, not at the array compare).
    """
    mx.random.seed(0)
    latent = mx.random.normal((1, 8, 8, 16))
    mx.eval(latent)
    weights = CONVERTED / "taef1_decoder.safetensors"
    z = Taef.from_kernel(get_kernel("zimage"), decoder_path=weights)
    t = Taef.from_kernel(get_kernel("taef1"), decoder_path=weights)
    assert get_kernel("zimage").latent.channels == 16
    out_z = np.array(z.decode(latent))
    out_t = np.array(t.decode(latent))
    assert np.array_equal(out_z, out_t)
    assert out_z.shape == (1, 64, 64, 3)
    assert out_z.min() >= 0.0  # decode() clips to [0,1]
    assert out_z.max() <= 1.0


def test_zimage_encode_bit_identical_to_taef1():
    """encode() reuses TAEF1's encoder weights, so it's byte-for-byte identical to TAEF1.

    Proves the inherited ZImage.encode() is correctly wired (same arch + weights). It does
    NOT claim parity with mflux's distinct Z-Image VAE encoder — that's out of scope for
    v0.4.0 (the validated path is decode/live-preview; see backlog 0047)."""
    mx.random.seed(0)
    image = mx.random.uniform(shape=(1, 64, 64, 3))
    mx.eval(image)
    dec = CONVERTED / "taef1_decoder.safetensors"
    enc = CONVERTED / "taef1_encoder.safetensors"
    z = Taef.from_kernel(get_kernel("zimage"), decoder_path=dec, encoder_path=enc)
    t = Taef.from_kernel(get_kernel("taef1"), decoder_path=dec, encoder_path=enc)
    assert np.array_equal(np.array(z.encode(image)), np.array(t.encode(image)))
