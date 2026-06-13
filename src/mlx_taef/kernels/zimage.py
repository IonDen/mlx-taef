"""Z-Image model kernel and its mflux latent unpack.

Z-Image's VAE shares FLUX.1's latent contract (scaling_factor=0.3611, shift_factor=0.1159,
spatial_scale=8, latent_channels=16), so this kernel reuses TAEF1's arch and weights — same
`TAESD2D` arch, same `madebyollin/taef1` `WeightSource` (one shared converted-cache entry).
"""

import mlx.core as mx

from mlx_taef.kernels._conversion import DiffusersRemap
from mlx_taef.kernels._types import (
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    UnpackContext,
    WeightSource,
)
from mlx_taef.kernels.flux import TAESD2D


def unpack_zimage_latent(latent: mx.array, ctx: UnpackContext) -> mx.array:
    """Unpack mflux's 4D Z-Image in-loop latent into NHWC (1, h, w, 16) for TAEF1.

    mflux's Z-Image in-loop latent is `(16, 1, h, w)` — channels-first, no batch, with a
    singleton temporal axis. This mirrors mflux `ZImageLatentCreator.unpack_latents`
    (`expand_dims(0)` -> `squeeze(2)` -> `(1,16,h,w)`) then NCHW->NHWC for TAEF1.
    `ctx.latent_height/width` are unused (the latent already carries its spatial dims) but
    kept for the uniform `(latent, UnpackContext)` binding signature.
    """
    if latent.ndim != 4 or latent.shape[0] != 16 or latent.shape[1] != 1:
        raise ValueError(
            f"Expected Z-Image in-loop latent (16, 1, h, w), got {tuple(latent.shape)}"
        )
    x = mx.expand_dims(latent, axis=0)  # (1, 16, 1, h, w)
    x = mx.squeeze(x, axis=2)  # (1, 16, h, w)
    return mx.transpose(x, (0, 2, 3, 1))  # (1, h, w, 16) NHWC for TAEF1


ZIMAGE = ModelKernel(
    name="zimage",
    arch=TAESD2D,
    conversion=DiffusersRemap(),
    latent=LatentSpec(channels=16),
    source=WeightSource(repo="madebyollin/taef1", filename="diffusion_pytorch_model.safetensors"),
    integration=MfluxBinding(
        mflux_models=("z-image", "z-image-turbo"), unpack=unpack_zimage_latent
    ),
    memory_cap_hint_gb=1,
)
