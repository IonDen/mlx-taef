"""unpack_qwen_latent matches mflux's own Qwen latent unpack (the binding oracle).

The arch parity fixtures test the decoder/encoder, NOT the mflux unpack — so this oracle is the
real guard for the "no denormalize" decision and the packed-latent reshape.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.kernels._types import UnpackContext
from mlx_taef.kernels.qwen import unpack_qwen_latent


def test_unpack_qwen_matches_mflux_qwen_latent_creator():
    pytest.importorskip("mflux")
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

    # arange + exact equality: the unpack is a pure permutation, so any wrong axis order is an
    # exact mismatch (a tolerance on random data could hide a swap of near-equal elements).
    lh, lw = 4, 6  # non-square, H,W>1
    packed = mx.arange(lh * lw * 64).reshape(1, lh * lw, 64).astype(mx.float32)
    ours = np.array(unpack_qwen_latent(packed, UnpackContext(latent_height=lh, latent_width=lw)))
    ref_nchw = QwenLatentCreator.unpack_latents(packed, height=lh * 16, width=lw * 16)
    ref_nhwc = np.array(mx.transpose(ref_nchw, (0, 2, 3, 1)))
    assert ours.shape == (1, lh * 2, lw * 2, 16)
    assert np.array_equal(ours, ref_nhwc)


def test_unpack_qwen_rejects_non_64_channels():
    with pytest.raises(ValueError, match="64-channel"):
        unpack_qwen_latent(mx.zeros((1, 24, 16)), UnpackContext(latent_height=4, latent_width=6))


def test_unpack_qwen_rejects_sequence_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"expected 6.*got 5"):
        unpack_qwen_latent(mx.zeros((1, 5, 64)), UnpackContext(latent_height=2, latent_width=3))
