"""Z-Image kernel: unpack value/rejection + offline equivalence to TAEF1."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_taef.kernels import UnpackContext
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
