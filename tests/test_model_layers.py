"""Tests for individual MLX layer modules in mlx_taef.model."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_taef.model import Block, Clamp, make_conv


def test_clamp_passes_zero_unchanged():
    layer = Clamp()
    x = mx.zeros((1, 4, 4, 8))
    out = layer(x)
    # atol=1e-6: fp32 zero should be exactly zero; 1e-6 guards against any
    # spurious subnormal flush-to-zero behavior on Metal.
    assert np.allclose(np.array(out), 0.0, atol=1e-6)


def test_clamp_saturates_large_positives_to_near_three():
    layer = Clamp()
    x = mx.full((1, 4, 4, 8), 1e6)
    out = np.array(layer(x))
    # tanh(1e6/3) * 3 -> 3.0 within fp32 precision
    assert np.allclose(out, 3.0, atol=1e-4)


def test_clamp_saturates_large_negatives_to_near_negative_three():
    layer = Clamp()
    x = mx.full((1, 4, 4, 8), -1e6)
    out = np.array(layer(x))
    assert np.allclose(out, -3.0, atol=1e-4)


def test_clamp_matches_pytorch_reference():
    """Tier 1 parity: MLX Clamp ≡ PyTorch tanh(x/3)*3 within fp32 tolerance."""
    torch = pytest.importorskip("torch")
    layer = Clamp()
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 8, 8, 16)).astype(np.float32) * 5.0
    mlx_out = np.array(layer(mx.array(x_np)))
    torch_out = (torch.tanh(torch.from_numpy(x_np) / 3) * 3).numpy()
    # atol=1e-5: fp32 vs fp32 — standard tolerance for identical math on CPU/Metal.
    # See skill §3: "fp32 layer vs NumPy reference: atol=1e-5, rtol=1e-5".
    assert np.allclose(mlx_out, torch_out, atol=1e-5)


def test_make_conv_default_is_3x3_padding_1():
    conv = make_conv(8, 16)
    # MLX NHWC weight shape: (out=16, kH=3, kW=3, in=8)
    assert conv.weight.shape == (16, 3, 3, 8)


def test_make_conv_stride_2():
    conv = make_conv(8, 16, stride=2)
    # MLX nn.Conv2d stores stride as (stride_h, stride_w) tuple
    assert conv.stride == (2, 2)


def test_make_conv_no_bias():
    conv = make_conv(8, 16, bias=False)
    # MLX removes the bias attribute entirely when bias=False (does not set it to None)
    assert not hasattr(conv, "bias")


def test_make_conv_matches_pytorch_at_fixed_weights():
    """Tier 1 parity: MLX Conv2d with NHWC-transposed weights produces equal output to PyTorch."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F  # noqa: N812

    rng = np.random.default_rng(0)
    weight_nchw = rng.standard_normal((16, 8, 3, 3)).astype(np.float32)
    bias = rng.standard_normal(16).astype(np.float32)
    x_nchw = rng.standard_normal((1, 8, 32, 32)).astype(np.float32)

    # PyTorch forward (NCHW)
    torch_out = F.conv2d(
        torch.from_numpy(x_nchw),
        torch.from_numpy(weight_nchw),
        bias=torch.from_numpy(bias),
        padding=1,
    ).numpy()

    # MLX forward (NHWC after weight transpose)
    mlx_conv = make_conv(8, 16)
    weight_nhwc = np.transpose(weight_nchw, (0, 2, 3, 1)).copy()
    mlx_conv.weight = mx.array(weight_nhwc)
    mlx_conv.bias = mx.array(bias)
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    mlx_out_nhwc = np.array(mlx_conv(mx.array(x_nhwc)))
    mlx_out_nchw = np.transpose(mlx_out_nhwc, (0, 3, 1, 2))

    # atol=1e-4: slightly looser than 1e-5 because of NCHW→NHWC transposition
    # and accumulated rounding across the padding+conv op on Metal vs PyTorch CPU.
    assert np.allclose(mlx_out_nchw, torch_out, atol=1e-4)


def test_block_same_in_out_uses_identity_skip():
    b = Block(64, 64)
    # skip should be Identity (no learnable weight)
    assert not hasattr(b.skip, "weight")


def test_block_different_in_out_uses_conv1x1_skip():
    b = Block(64, 128)
    assert hasattr(b.skip, "weight")
    # MLX NHWC 1x1 conv weight shape: (out=128, kH=1, kW=1, in=64)
    assert b.skip.weight.shape == (128, 1, 1, 64)


def test_block_shape_preserved_when_same_channels():
    b = Block(64, 64)
    x = mx.zeros((1, 16, 16, 64))
    out = b(x)
    assert out.shape == (1, 16, 16, 64)


def test_block_shape_changes_channels_when_different():
    b = Block(64, 128)
    x = mx.zeros((1, 16, 16, 64))
    out = b(x)
    assert out.shape == (1, 16, 16, 128)


def test_block_parity_with_pytorch_reference():
    """Tier 1 parity: MLX Block (no midblock_gn) ≡ PyTorch Block."""
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    def t_conv(n_in: int, n_out: int) -> tnn.Conv2d:
        return tnn.Conv2d(n_in, n_out, 3, padding=1)

    class TBlock(tnn.Module):
        def __init__(self, n_in: int, n_out: int) -> None:
            super().__init__()
            self.conv = tnn.Sequential(
                t_conv(n_in, n_out),
                tnn.ReLU(),
                t_conv(n_out, n_out),
                tnn.ReLU(),
                t_conv(n_out, n_out),
            )
            self.skip = tnn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else tnn.Identity()
            self.fuse = tnn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fuse(self.conv(x) + self.skip(x))

    torch.manual_seed(0)
    tb = TBlock(8, 16)
    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((1, 8, 16, 16)).astype(np.float32)
    with torch.no_grad():
        t_out = tb(torch.from_numpy(x_nchw)).numpy()

    # Mirror weights into MLX block
    mb = Block(8, 16)
    # conv is Sequential[Conv, ReLU, Conv, ReLU, Conv]; indices 0, 2, 4 are convs.
    # MLX's nn.Sequential exposes layers as `.layers[i]`.
    for src_idx, dst_idx in [(0, 0), (2, 2), (4, 4)]:
        src = tb.conv[src_idx]
        mb.conv.layers[dst_idx].weight = mx.array(
            np.transpose(src.weight.detach().numpy(), (0, 2, 3, 1)).copy()
        )
        mb.conv.layers[dst_idx].bias = mx.array(src.bias.detach().numpy())
    # skip is a 1x1 Conv2d (since n_in != n_out)
    mb.skip.weight = mx.array(np.transpose(tb.skip.weight.detach().numpy(), (0, 2, 3, 1)).copy())

    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    m_out_nhwc = np.array(mb(mx.array(x_nhwc)))
    m_out_nchw = np.transpose(m_out_nhwc, (0, 3, 1, 2))

    # atol=1e-4: three-conv chain with ReLU accumulates more rounding than
    # a single conv; Metal fp32 arithmetic may differ from PyTorch CPU fp32.
    assert np.allclose(m_out_nchw, t_out, atol=1e-4)


def test_block_with_midblock_gn_has_pool_branch():
    b = Block(64, 64, use_midblock_gn=True)
    assert b.pool is not None


def test_block_with_midblock_gn_shape_preserved():
    b = Block(64, 64, use_midblock_gn=True)
    x = mx.zeros((1, 8, 8, 64))
    out = b(x)
    assert out.shape == (1, 8, 8, 64)


def test_block_midblock_gn_parity_with_pytorch():
    """Tier 1 parity: MLX Block(use_midblock_gn=True) ≡ PyTorch reference.

    Requires nn.GroupNorm(pytorch_compatible=True) on the MLX side — without
    that flag, MLX uses a different group-channel ordering that produces
    numerically different output even on bit-identical weights.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    def t_conv(n_in: int, n_out: int) -> tnn.Conv2d:
        return tnn.Conv2d(n_in, n_out, 3, padding=1)

    class TBlock(tnn.Module):
        def __init__(self, n_in: int, n_out: int) -> None:
            super().__init__()
            self.conv = tnn.Sequential(
                t_conv(n_in, n_out),
                tnn.ReLU(),
                t_conv(n_out, n_out),
                tnn.ReLU(),
                t_conv(n_out, n_out),
            )
            self.skip = tnn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else tnn.Identity()
            self.fuse = tnn.ReLU()
            n_gn = n_in * 4
            self.pool = tnn.Sequential(
                tnn.Conv2d(n_in, n_gn, 1, bias=False),
                tnn.GroupNorm(4, n_gn),
                tnn.ReLU(),
                tnn.Conv2d(n_gn, n_in, 1, bias=False),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pool(x)
            return self.fuse(self.conv(x) + self.skip(x))

    torch.manual_seed(1)
    tb = TBlock(16, 16)
    rng = np.random.default_rng(1)
    x_nchw = rng.standard_normal((1, 16, 8, 8)).astype(np.float32)
    with torch.no_grad():
        t_out = tb(torch.from_numpy(x_nchw)).numpy()

    mb = Block(16, 16, use_midblock_gn=True)
    # Copy conv chain weights (indices 0, 2, 4 in the Sequential)
    for src_idx, dst_idx in [(0, 0), (2, 2), (4, 4)]:
        src = tb.conv[src_idx]
        mb.conv.layers[dst_idx].weight = mx.array(
            np.transpose(src.weight.detach().numpy(), (0, 2, 3, 1)).copy()
        )
        mb.conv.layers[dst_idx].bias = mx.array(src.bias.detach().numpy())
    # Copy pool weights (Sequential[Conv1x1, GroupNorm, ReLU, Conv1x1])
    assert mb.pool is not None
    mb.pool.layers[0].weight = mx.array(
        np.transpose(tb.pool[0].weight.detach().numpy(), (0, 2, 3, 1)).copy()
    )
    mb.pool.layers[1].weight = mx.array(tb.pool[1].weight.detach().numpy())
    mb.pool.layers[1].bias = mx.array(tb.pool[1].bias.detach().numpy())
    mb.pool.layers[3].weight = mx.array(
        np.transpose(tb.pool[3].weight.detach().numpy(), (0, 2, 3, 1)).copy()
    )

    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    m_out_nhwc = np.array(mb(mx.array(x_nhwc)))
    m_out_nchw = np.transpose(m_out_nhwc, (0, 3, 1, 2))

    # atol=1e-4: GroupNorm on Metal adds one more fp32 rounding stage vs PyTorch CPU.
    # pytorch_compatible=True on nn.GroupNorm is required; without it, MLX uses a
    # different group-channel ordering that produces larger divergence (~1e-2).
    assert np.allclose(m_out_nchw, t_out, atol=1e-4), (
        f"Block midblock_gn parity failed. "
        f"Max diff: {np.abs(m_out_nchw - t_out).max():.6f}. "
        f"Check pytorch_compatible=True on nn.GroupNorm."
    )


def test_mlx_upsample_doubles_spatial_dims_for_nhwc():
    """Verify nn.Upsample default operates on NHWC spatial axes."""
    up = nn.Upsample(scale_factor=2, mode="nearest")
    x = mx.zeros((1, 4, 4, 8))
    out = up(x)
    assert out.shape == (1, 8, 8, 8), f"Expected (1, 8, 8, 8), got {out.shape}"


def test_mlx_upsample_nearest_neighbor_behavior():
    """Single nonzero pixel must spread to a 2x2 block."""
    up = nn.Upsample(scale_factor=2, mode="nearest")
    x = np.zeros((1, 2, 2, 1), dtype=np.float32)
    x[0, 0, 0, 0] = 5.0
    out = np.array(up(mx.array(x)))
    assert np.allclose(out[0, 0:2, 0:2, 0], 5.0)
    assert out.sum() == 5.0 * 4


def test_mlx_upsample_matches_pytorch_reference():
    """Parity: MLX nn.Upsample(scale_factor=2, mode='nearest') ≡ PyTorch Upsample(scale_factor=2)."""
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((1, 4, 3, 3)).astype(np.float32)
    t_out = tnn.Upsample(scale_factor=2, mode="nearest")(torch.from_numpy(x_nchw)).numpy()
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    m_out_nhwc = np.array(nn.Upsample(scale_factor=2, mode="nearest")(mx.array(x_nhwc)))
    m_out_nchw = np.transpose(m_out_nhwc, (0, 3, 1, 2))
    # atol=1e-6: nearest-neighbor upsample is exact integer indexing; should be
    # bit-for-bit identical to PyTorch, modulo fp32 copy overhead.
    assert np.allclose(m_out_nchw, t_out, atol=1e-6)
