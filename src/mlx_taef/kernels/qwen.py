"""Qwen-Image (taew2.1) model kernel and its mflux latent unpack.

Decodes Qwen-Image / Qwen-Image-Edit latents (Wan 2.1 VAE, 16 channels) via madebyollin's
taew2.1 tiny autoencoder ported to MLX (the `taehv` arch). The canonical weights are GitHub-only;
the kernel pins a sha256-verified Hugging Face re-host.
"""

import mlx.core as mx

from mlx_taef.kernels._conversion import TaehvCombined
from mlx_taef.kernels._types import (
    ArchSpec,
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    UnpackContext,
    WeightSource,
)


def unpack_qwen_latent(latent: mx.array, ctx: UnpackContext) -> mx.array:
    """Unpack mflux's packed Qwen-Image latent into NHWC (B, lh*2, lw*2, 16) for taew2.1.

    Qwen-Image's in-loop latent is packed identically to FLUX.1 — mflux's `QwenLatentCreator`
    delegates to `FluxLatentCreator` — so this mirrors `unpack_flux1_latent`. NO denormalize:
    taew2.1 consumes the normalized diffusion latent directly (the Wan per-channel mean/std is
    baked into its weights), exactly as TAEF1 consumes the raw FLUX latent.
    """
    b, _, c = latent.shape
    if c != 64:
        raise ValueError(f"Expected 64-channel packed Qwen-Image latent, got {c}")
    lh, lw = ctx.latent_height, ctx.latent_width
    x = latent.reshape(b, lh, lw, 16, 2, 2)
    x = x.transpose(0, 3, 1, 4, 2, 5)  # (b, 16, lh, 2, lw, 2)
    x = x.reshape(b, 16, lh * 2, lw * 2)
    return x.transpose(0, 2, 3, 1)  # NHWC for taew2.1


QWEN_IMAGE = ModelKernel(
    name="qwen-image",
    arch=ArchSpec(name="taehv"),
    conversion=TaehvCombined(),
    latent=LatentSpec(channels=16),
    # Canonical taew2.1 weights are published only on GitHub (madebyollin/taehv); this is the
    # sha256-verified Hugging Face re-host. sha256 is the integrity pin; `revision` is set to the
    # repo's commit once the re-host exists.
    source=WeightSource(
        repo="IonDen/taew2.1",
        filename="taew2_1.safetensors",
        sha256="04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f",
    ),
    integration=MfluxBinding(
        mflux_models=("qwen-image", "qwen-image-edit"),
        unpack=unpack_qwen_latent,
        packed_latent_downscale=16,
    ),
    memory_cap_hint_gb=1,
)
