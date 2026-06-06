"""MLX layer modules for the TAESD-family of tiny autoencoders.

Layers are direct ports of the upstream PyTorch implementations at
https://github.com/madebyollin/taesd (MIT licensed).

All public layers accept and return NHWC `mx.array` tensors.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_taef.variants import TaesdVariantConfig


class Clamp(nn.Module):  # type: ignore[misc,name-defined]
    """Soft clamp via `tanh(x / 3) * 3` — bounds output to [-3, 3].

    Direct port of upstream `taesd.Clamp`. Used at the start of the decoder
    to bound latent magnitudes before the first conv.
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the soft clamp.

        Args:
            x: input tensor of any shape.

        Returns:
            Element-wise `tanh(x / 3) * 3` of the same shape.
        """
        return mx.tanh(x / 3.0) * 3.0


def make_conv(n_in: int, n_out: int, *, stride: int = 1, bias: bool = True) -> "nn.Conv2d":  # type: ignore[name-defined]
    """3x3 Conv2d with padding=1 — matches upstream `taesd.conv(n_in, n_out, **kwargs)`.

    Args:
        n_in: input channel count.
        n_out: output channel count.
        stride: spatial stride for the convolution (default 1).
        bias: whether to include a learned bias term (default True).

    Returns:
        Configured `mlx.nn.Conv2d` with kernel_size=3, padding=1.
    """
    return nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=bias)  # type: ignore[attr-defined]


class _Identity(nn.Module):  # type: ignore[misc,name-defined]
    """No-op layer used as the skip connection when in/out channels match."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class Block(nn.Module):  # type: ignore[misc,name-defined]
    """Residual block: `ReLU(Conv -> ReLU -> Conv -> ReLU -> Conv + skip)`.

    Direct port of upstream `taesd.Block`. Optionally adds a parallel
    `midblock_gn` pool branch (used by the flux_2 TAEF2 variant).

    Args:
        n_in: input channel count.
        n_out: output channel count.
        use_midblock_gn: when True, prepend `x = x + pool(x)` where `pool` is
            a 1x1 conv expand -> GroupNorm -> ReLU -> 1x1 conv contract chain.
            Used by TAEF2's flux_2 arch_variant. Default False.
    """

    def __init__(self, n_in: int, n_out: int, *, use_midblock_gn: bool = False) -> None:
        """Initialise Block layers.

        Args:
            n_in: input channel count.
            n_out: output channel count.
            use_midblock_gn: when True, prepend the midblock GroupNorm pool
                branch before the main conv path. Default False.
        """
        super().__init__()
        self.conv = nn.Sequential(  # type: ignore[attr-defined]
            make_conv(n_in, n_out),
            nn.ReLU(),  # type: ignore[attr-defined]
            make_conv(n_out, n_out),
            nn.ReLU(),  # type: ignore[attr-defined]
            make_conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, kernel_size=1, bias=False) if n_in != n_out else _Identity()  # type: ignore[attr-defined]
        )
        self.fuse = nn.ReLU()  # type: ignore[attr-defined]
        self.pool = None
        if use_midblock_gn:
            n_gn = n_in * 4
            # CRITICAL: pytorch_compatible=True is required for parity. MLX
            # defaults to a different group-channel ordering than PyTorch's
            # nn.GroupNorm.
            self.pool = nn.Sequential(  # type: ignore[attr-defined]
                nn.Conv2d(n_in, n_gn, kernel_size=1, bias=False),  # type: ignore[attr-defined]
                nn.GroupNorm(num_groups=4, dims=n_gn, pytorch_compatible=True),  # type: ignore[attr-defined]
                nn.ReLU(),  # type: ignore[attr-defined]
                nn.Conv2d(n_gn, n_in, kernel_size=1, bias=False),  # type: ignore[attr-defined]
            )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the residual block.

        Args:
            x: NHWC input tensor with `n_in` channels.

        Returns:
            NHWC output tensor with `n_out` channels.
        """
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))  # type: ignore[no-any-return]


def make_decoder(config: TaesdVariantConfig) -> nn.Sequential:  # type: ignore[name-defined]
    """Build the decoder for a TAESD-family variant.

    Mirrors upstream `taesd.Decoder(latent_channels, use_midblock_gn)`:
    Clamp -> Conv -> ReLU -> Block x3 -> Up -> Conv -> Block x3 -> Up -> Conv ->
    Block x3 -> Up -> Conv -> Block -> Conv.

    For arch_variant="flux_2" (TAEF2), the first three Blocks have
    `use_midblock_gn=True`; the rest are vanilla.

    Args:
        config: variant configuration that selects `latent_channels` and
            whether the first three Blocks use midblock_gn.

    Returns:
        `nn.Sequential` decoder ready for weight loading.
    """
    from mlx_taef.kernels._arch import build_arch

    return build_arch(
        "taesd2d",
        role="decoder",
        latent_channels=config.latent_channels,
        midblock_gn=config.use_midblock_gn,
    )


def make_encoder(config: TaesdVariantConfig) -> nn.Sequential:  # type: ignore[name-defined]
    """Build the encoder for a TAESD-family variant.

    Mirrors upstream `taesd.Encoder(latent_channels, use_midblock_gn)`:
    Conv -> Block -> (StridedConv -> Block x3) x3 -> Conv.
    Three strided convs each halve spatial dims (8x downsample total).

    For arch_variant="flux_2" (TAEF2), the LAST three Blocks use `use_midblock_gn=True`.

    Args:
        config: variant configuration that selects `latent_channels` and `arch_variant`.

    Returns:
        `nn.Sequential` encoder ready for weight loading.
    """
    from mlx_taef.kernels._arch import build_arch

    return build_arch(
        "taesd2d",
        role="encoder",
        latent_channels=config.latent_channels,
        midblock_gn=config.use_midblock_gn,
    )


__all__ = ["Block", "Clamp", "make_conv", "make_decoder", "make_encoder"]
