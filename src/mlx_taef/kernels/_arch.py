"""Named architecture builders. A kernel's `ArchSpec.name` selects one of these.

`taesd2d` is the shared TAESD-family 2D image arch (TAESD/TAESDXL/TAEF1/TAEF2). Builder
knobs (latent_channels, midblock_gn) are passed at build time — they do NOT live on the
shared `ArchSpec` record.
"""

from collections.abc import Callable

import mlx.nn as nn

from mlx_taef.model import Block, Clamp, make_conv


def _build_taesd2d_decoder(latent_channels: int, *, midblock_gn: bool) -> "nn.Sequential":  # type: ignore[name-defined]
    layers: list[nn.Module] = [  # type: ignore[name-defined]
        Clamp(),
        make_conv(latent_channels, 64),
        nn.ReLU(),  # type: ignore[attr-defined]
        Block(64, 64, use_midblock_gn=midblock_gn),
        Block(64, 64, use_midblock_gn=midblock_gn),
        Block(64, 64, use_midblock_gn=midblock_gn),
        nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
        make_conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
        make_conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2, mode="nearest"),  # type: ignore[attr-defined]
        make_conv(64, 64, bias=False),
        Block(64, 64),
        make_conv(64, 3),
    ]
    return nn.Sequential(*layers)  # type: ignore[attr-defined]


def _build_taesd2d_encoder(latent_channels: int, *, midblock_gn: bool) -> "nn.Sequential":  # type: ignore[name-defined]
    layers: list[nn.Module] = [  # type: ignore[name-defined]
        make_conv(3, 64),
        Block(64, 64),
        make_conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        make_conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        make_conv(64, 64, stride=2, bias=False),
        Block(64, 64, use_midblock_gn=midblock_gn),
        Block(64, 64, use_midblock_gn=midblock_gn),
        Block(64, 64, use_midblock_gn=midblock_gn),
        make_conv(64, latent_channels),
    ]
    return nn.Sequential(*layers)  # type: ignore[attr-defined]


def _build_taehv_decoder(latent_channels: int, *, midblock_gn: bool) -> "nn.Sequential":  # type: ignore[name-defined]
    """Build the taew2.1 (taehv) decoder. `midblock_gn` is accepted for the uniform call site but unused."""
    from mlx_taef.kernels._taehv import TaehvDecoder

    return TaehvDecoder(latent_channels=latent_channels)


def _build_taehv_encoder(latent_channels: int, *, midblock_gn: bool) -> "nn.Sequential":  # type: ignore[name-defined]
    """Build the taew2.1 (taehv) encoder. `midblock_gn` is accepted for the uniform call site but unused."""
    from mlx_taef.kernels._taehv import TaehvEncoder

    return TaehvEncoder(latent_channels=latent_channels)


ARCH_BUILDERS: dict[str, dict[str, Callable[..., nn.Sequential]]] = {  # type: ignore[name-defined]
    "taesd2d": {"decoder": _build_taesd2d_decoder, "encoder": _build_taesd2d_encoder},
    "taehv": {"decoder": _build_taehv_decoder, "encoder": _build_taehv_encoder},
}


def build_arch(
    arch_name: str, *, role: str, latent_channels: int, midblock_gn: bool
) -> "nn.Sequential":  # type: ignore[name-defined]
    """Build the encoder/decoder for `arch_name` with the given channel count + knobs."""
    try:
        builder = ARCH_BUILDERS[arch_name][role]
    except KeyError as e:
        raise KeyError(f"unknown arch/role: {arch_name!r}/{role!r}") from e
    return builder(latent_channels, midblock_gn=midblock_gn)
