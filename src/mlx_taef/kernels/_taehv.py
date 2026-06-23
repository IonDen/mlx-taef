"""Pure-MLX port of madebyollin taehv blocks (Tiny AutoEncoder for Wan 2.1 / Qwen-Image).

Direct port of the upstream PyTorch blocks at https://github.com/madebyollin/taehv (MIT).
All layers are NHWC. The temporal axis is folded into the batch dim (NT) by the decoder/
encoder driver; the blocks here are 2D and operate on `(NT, H, W, C)`.

The `TGrow`/`TPool` time reshapes are NOT a literal transcription of upstream's NCHW
`reshape(-1, C, H, W)` — in NHWC the channel axis is last, so a literal reshape would scramble
spatial positions with frames. The transpose-based recipes below reproduce the NCHW row-major
grouping exactly (verified against an NCHW reference at H,W>1 in tests/test_taehv_blocks.py).
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_taef.model import _Identity, make_conv


class MemBlock(nn.Module):  # type: ignore[misc,name-defined]
    """Recurrent block: `ReLU(conv(cat([x, past])) + skip(x))`.

    Port of upstream `taehv.MemBlock`. The conv path's first layer takes `2 * n_in` channels
    because the current frame `x` is concatenated with the previous frame's memory `past` on
    the channel (last) axis. taew2.1's MemBlocks are all square (`n_in == n_out`), so `skip` is
    `_Identity`, but the conditional is kept for faithfulness.
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        """Build the MemBlock layers.

        Args:
            n_in: input channel count (per frame).
            n_out: output channel count.
        """
        super().__init__()
        self.conv = nn.Sequential(  # type: ignore[attr-defined]
            make_conv(n_in * 2, n_out),
            nn.ReLU(),  # type: ignore[attr-defined]
            make_conv(n_out, n_out),
            nn.ReLU(),  # type: ignore[attr-defined]
            make_conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, kernel_size=1, bias=False) if n_in != n_out else _Identity()  # type: ignore[attr-defined]
        )
        self.fuse = nn.ReLU()  # type: ignore[attr-defined]

    def __call__(self, x: mx.array, past: mx.array) -> mx.array:
        """Apply the block.

        Args:
            x: NHWC `(NT, H, W, n_in)` current-frame activations.
            past: NHWC `(NT, H, W, n_in)` previous-frame memory (zeros for the first frame).

        Returns:
            NHWC `(NT, H, W, n_out)` activations.
        """
        h = mx.concatenate([x, past], axis=-1)
        return self.fuse(self.conv(h) + self.skip(x))  # type: ignore[no-any-return]


class TGrow(nn.Module):  # type: ignore[misc,name-defined]
    """Temporal upsample: 1x1 conv `C -> C*stride`, then split the stride off into time.

    Port of upstream `taehv.TGrow`. Upstream (NCHW) does `conv` then `reshape(-1, C, H, W)`,
    growing `NT -> NT*stride`. In NHWC the equivalent split needs a transpose (see module
    docstring).
    """

    def __init__(self, n_f: int, stride: int) -> None:
        """Build the TGrow layer.

        Args:
            n_f: channel count.
            stride: temporal upsample factor (1 = no temporal growth).
        """
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, kernel_size=1, bias=False)  # type: ignore[attr-defined]

    def _reshape_time(self, y: mx.array) -> mx.array:
        """Split the post-conv channel axis `C*stride` into time: `(NT,H,W,C*s) -> (NT*s,H,W,C)`."""
        nt, h, w, cs = y.shape
        c = cs // self.stride
        y = y.reshape(nt, h, w, self.stride, c)
        y = y.transpose(0, 3, 1, 2, 4)  # (NT, s, H, W, C)
        return y.reshape(nt * self.stride, h, w, c)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 1x1 conv then the time split. Input/output NHWC."""
        return self._reshape_time(self.conv(x))


class TPool(nn.Module):  # type: ignore[misc,name-defined]
    """Temporal downsample: group `stride` frames into the channel axis, then 1x1 conv.

    Port of upstream `taehv.TPool`. Upstream (NCHW) does `reshape(-1, stride*C, H, W)` then
    `conv`, shrinking `NT -> NT/stride`. In NHWC the equivalent group needs a transpose.
    """

    def __init__(self, n_f: int, stride: int) -> None:
        """Build the TPool layer.

        Args:
            n_f: channel count.
            stride: temporal downsample factor (1 = no temporal pooling).
        """
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, kernel_size=1, bias=False)  # type: ignore[attr-defined]

    def _reshape_time(self, x: mx.array) -> mx.array:
        """Group `stride` frames into the channel axis: `(NT,H,W,C) -> (NT/s,H,W,s*C)`."""
        nt, h, w, c = x.shape
        s = self.stride
        x = x.reshape(nt // s, s, h, w, c)
        x = x.transpose(0, 2, 3, 1, 4)  # (NT/s, H, W, s, C)
        return x.reshape(nt // s, h, w, s * c)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the time group then 1x1 conv. Input/output NHWC."""
        return self.conv(self._reshape_time(x))  # type: ignore[no-any-return]
