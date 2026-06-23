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

from mlx_taef.model import Clamp, _Identity, make_conv


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


def _shift_memory(x: mx.array, n: int) -> mx.array:
    """Previous-frame memory for a MemBlock: prepend a zero frame on time, drop the last.

    Mirrors upstream `pad(_x, (...,1,0))[:, :T]`. `T = NT // n` is recomputed from the current
    tensor each call, because intervening TGrows grow NT.
    """
    nt, h, w, c = x.shape
    t = nt // n
    g = x.reshape(n, t, h, w, c)
    zero = mx.zeros((n, 1, h, w, c), dtype=x.dtype)
    g = mx.concatenate([zero, g], axis=1)[:, :t]
    return g.reshape(nt, h, w, c)


def _run_memblocks(layers: object, x: mx.array, n: int) -> mx.array:
    """Iterate the Sequential's children with the memory-threaded driver (time folded into NT)."""
    for layer in layers:  # type: ignore[attr-defined]
        x = layer(x, _shift_memory(x, n)) if isinstance(layer, MemBlock) else layer(x)
    return x


class TaehvDecoder(nn.Sequential):  # type: ignore[misc,name-defined]
    """taew2.1 decoder as an MLX nn.Sequential (param keys `layers.N...`) with a custom driver.

    Child order mirrors upstream exactly (incl. paramless Clamp/ReLU/Upsample) so indices align
    for weight conversion. `__call__` runs the memory-threaded driver, not a feed-forward chain.
    For a single image (T=1) the temporal TGrows grow NT to `T_UPSCALE`; the leading
    `T_UPSCALE-1` output frames are trimmed, leaving one RGB frame.
    """

    T_UPSCALE = 4  # two decoder TGrows with stride 2 (decoder_time_upscale=(False,True,True))

    def __init__(self, latent_channels: int = 16) -> None:
        """Build the decoder blocks for the given latent channel count."""
        super().__init__(
            Clamp(),
            make_conv(latent_channels, 256),
            nn.ReLU(),  # type: ignore[attr-defined]
            MemBlock(256, 256),
            MemBlock(256, 256),
            MemBlock(256, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
            TGrow(256, 1),
            make_conv(256, 128, bias=False),
            MemBlock(128, 128),
            MemBlock(128, 128),
            MemBlock(128, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
            TGrow(128, 2),
            make_conv(128, 64, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
            TGrow(64, 2),
            make_conv(64, 64, bias=False),
            nn.ReLU(),  # type: ignore[attr-defined]
            make_conv(64, 3),
        )

    def __call__(self, latent: mx.array) -> mx.array:
        """Decode an NHWC latent `(B,H,W,16)` (T=1) to RGB `(B,H*8,W*8,3)` in [0,1]."""
        n = latent.shape[0]
        x = _run_memblocks(self.layers, latent, n)
        nt = x.shape[0]
        t_out = nt // n
        x = x.reshape(n, t_out, *x.shape[1:])[:, self.T_UPSCALE - 1 :]
        x = x.reshape(-1, *x.shape[2:])
        return mx.clip(x, 0.0, 1.0)


class TaehvEncoder(nn.Sequential):  # type: ignore[misc,name-defined]
    """taew2.1 encoder as an MLX nn.Sequential with a custom driver.

    For a single image, the input frame is repeat-padded along time to `T_DOWNSCALE` before the
    encoder driver (the TPools require T divisible by their stride); the two stride-2 TPools then
    collapse it back to one latent frame.
    """

    T_DOWNSCALE = 4  # two encoder TPools with stride 2 (encoder_time_downscale=(True,True,False))

    def __init__(self, latent_channels: int = 16) -> None:
        """Build the encoder blocks for the given latent channel count."""
        super().__init__(
            make_conv(3, 64),
            nn.ReLU(),  # type: ignore[attr-defined]
            TPool(64, 2),
            make_conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            make_conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            make_conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            make_conv(64, latent_channels),
        )

    def __call__(self, image: mx.array) -> mx.array:
        """Encode NHWC RGB `(B,H,W,3)` (T=1) to a latent `(B,H/8,W/8,16)`."""
        n = image.shape[0]
        x = mx.repeat(mx.expand_dims(image, axis=1), self.T_DOWNSCALE, axis=1)
        x = x.reshape(n * self.T_DOWNSCALE, *image.shape[1:])
        return _run_memblocks(self.layers, x, n)
