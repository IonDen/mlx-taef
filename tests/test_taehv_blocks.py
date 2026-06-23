"""Shape/structure tests for the MLX taehv blocks (MemBlock, TGrow, TPool).

The NHWC time reshapes are NOT a literal transcription of upstream's NCHW reshapes
(channel axis is last in NHWC), so TGrow/TPool are checked against an explicit NCHW
reference at H,W>1 — a 1x1-spatial check would pass even a wrong port.
"""

import mlx.core as mx
import numpy as np

from mlx_taef.kernels._taehv import MemBlock, TGrow, TPool


def test_memblock_uses_past_memory_and_outputs_n_out():
    # NHWC, H,W>1, time folded into batch (NT=2). taehv MemBlocks are square (n_in==n_out).
    # Distinct non-zero x/past so we verify `past` is actually concatenated + used — an all-zero
    # test only checks shape and can't see the concat axis.
    block = MemBlock(64, 64)
    mx.eval(block.parameters())
    x = mx.random.normal((2, 4, 5, 64), key=mx.random.key(0))
    past = mx.random.normal((2, 4, 5, 64), key=mx.random.key(1))
    out = block(x, past)
    assert out.shape == (2, 4, 5, 64)
    # Changing the memory changes the output -> past is genuinely wired into the conv.
    assert not mx.allclose(out, block(x, mx.zeros_like(past))).item()


def _ref_tgrow_nchw(x_nchw: np.ndarray, stride: int) -> np.ndarray:
    # upstream: conv output (NT, C*s, H, W) -> reshape(-1, C, H, W) == (NT*s, C, H, W) row-major.
    nt, cs, h, w = x_nchw.shape
    c = cs // stride
    return x_nchw.reshape(nt * stride, c, h, w)


def _ref_tpool_nchw(x_nchw: np.ndarray, stride: int) -> np.ndarray:
    # upstream: (NT, C, H, W) -> reshape(-1, stride*C, H, W) == (NT/s, s*C, H, W) row-major.
    nt, c, h, w = x_nchw.shape
    return x_nchw.reshape(nt // stride, stride * c, h, w)


def test_tgrow_reshape_matches_nchw_grouping_at_hw_gt_1():
    nt, c, h, w, s = 2, 3, 4, 5, 2  # H,W>1 essential
    post_conv_nchw = np.arange(nt * c * s * h * w, dtype=np.float32).reshape(nt, c * s, h, w)
    post_conv_nhwc = mx.array(np.transpose(post_conv_nchw, (0, 2, 3, 1)))  # (NT,H,W,C*s)
    grow = TGrow(c, s)
    out_nhwc = np.asarray(grow._reshape_time(post_conv_nhwc))
    out_nchw = np.transpose(out_nhwc, (0, 3, 1, 2))
    assert out_nchw.shape == (nt * s, c, h, w)
    assert np.array_equal(out_nchw, _ref_tgrow_nchw(post_conv_nchw, s))


def test_tpool_reshape_matches_nchw_grouping_at_hw_gt_1():
    nt, c, h, w, s = 4, 3, 4, 5, 2
    pre_conv_nchw = np.arange(nt * c * h * w, dtype=np.float32).reshape(nt, c, h, w)
    pre_conv_nhwc = mx.array(np.transpose(pre_conv_nchw, (0, 2, 3, 1)))  # (NT,H,W,C)
    pool = TPool(c, s)
    out_nhwc = np.asarray(pool._reshape_time(pre_conv_nhwc))
    out_nchw = np.transpose(out_nhwc, (0, 3, 1, 2))
    assert out_nchw.shape == (nt // s, s * c, h, w)
    assert np.array_equal(out_nchw, _ref_tpool_nchw(pre_conv_nchw, s))
