"""FLUX.1 / FLUX.2 model kernels and their mflux latent unpacks."""

import mlx.core as mx

from mlx_taef.kernels._conversion import DiffusersRemap
from mlx_taef.kernels._types import (
    ArchSpec,
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    UnpackContext,
    WeightSource,
)

TAESD2D = ArchSpec(name="taesd2d")


def unpack_flux1_latent(latent: mx.array, ctx: UnpackContext) -> mx.array:
    """Unpack mflux's packed FLUX.1 in-loop latent into NHWC (B, lh*2, lw*2, 16) for TAEF1.

    mflux's in-loop FLUX.1 latent is PACKED `(B, lh*lw, 64)` (the denoiser operates on the
    packed form; mflux only unpacks after the loop). The pre-refactor callback's
    `shape[1] == 16` check never matched and fed the packed latent straight to TAEF1 -> a
    broken preview. This mirrors mflux `FluxLatentCreator.unpack_latents` EXACTLY
    (reshape `(1, lh, lw, 16, 2, 2)` -> transpose `(0,3,1,4,2,5)` -> `(1,16,lh*2,lw*2)`),
    then NCHW->NHWC for TAEF1.
    """
    b, _, c = latent.shape
    if c != 64:
        raise ValueError(f"Expected 64-channel packed FLUX.1 latent, got {c}")
    lh, lw = ctx.latent_height, ctx.latent_width
    x = latent.reshape(b, lh, lw, 16, 2, 2)
    x = x.transpose(0, 3, 1, 4, 2, 5)  # (b, 16, lh, 2, lw, 2)
    x = x.reshape(b, 16, lh * 2, lw * 2)
    return x.transpose(0, 2, 3, 1)  # NHWC for TAEF1


def unpack_flux2_latent(latent: mx.array, ctx: UnpackContext) -> mx.array:
    """Unpack mflux's packed FLUX.2 latent into NHWC (B, lh*2, lw*2, 32) for TAEF2.

    BN denormalize (128-ch stats) + unpatchify + NCHW->NHWC. BN stats come from ctx; absent
    -> identity BN. Transpose order matches the shipped pre-refactor unpack_flux2_latent.
    """
    b, _, c = latent.shape
    if c != 128:
        raise ValueError(f"Expected 128-channel packed FLUX.2 latent, got {c}")
    lh, lw = ctx.latent_height, ctx.latent_width
    latents = latent.reshape(b, lh, lw, c).transpose(0, 3, 1, 2)
    if ctx.bn_mean is not None and ctx.bn_var is not None:
        mean = ctx.bn_mean.reshape(1, -1, 1, 1)
        std = mx.sqrt(ctx.bn_var.reshape(1, -1, 1, 1) + ctx.bn_eps)
        latents = latents * std + mean
    batch, _, h, w = latents.shape
    latents = latents.reshape(batch, 32, 2, 2, h, w).transpose(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch, 32, h * 2, w * 2)
    return latents.transpose(0, 2, 3, 1)


TAEF1 = ModelKernel(
    name="taef1",
    arch=TAESD2D,
    conversion=DiffusersRemap(),
    latent=LatentSpec(channels=16),
    source=WeightSource(repo="madebyollin/taef1", filename="diffusion_pytorch_model.safetensors"),
    integration=MfluxBinding(
        mflux_models=("flux1", "flux-dev", "flux-schnell"),
        unpack=unpack_flux1_latent,
        packed_latent_downscale=16,
    ),
    memory_cap_hint_gb=1,
)

TAEF2 = ModelKernel(
    name="taef2",
    arch=TAESD2D,
    conversion=DiffusersRemap(),
    latent=LatentSpec(channels=32),
    source=WeightSource(repo="madebyollin/taef2", filename="taef2.safetensors"),
    integration=MfluxBinding(
        mflux_models=("flux2", "flux2-klein"),
        unpack=unpack_flux2_latent,
        packed_latent_downscale=16,
    ),
    memory_cap_hint_gb=2,
)
