"""Krea 2 model kernel and its mflux latent unpack.

Krea 2 generates on the Qwen-Image stack with the Qwen-Image (Wan 2.1) VAE — same
16-channel latent contract — so this kernel reuses the qwen-image kernel's taew2.1
weights (`ionden/taew2.1`, one shared converted-cache entry).

Verified against installed mflux 0.18.1: `mflux/models/krea2/latent_creator/
krea2_latent_creator.py`'s `Krea2LatentCreator.create_noise` produces the in-loop latent
as `(1, 16, height // 8, width // 8)` (4-D NCHW), and its `pack_latents`/`unpack_latents`
are both identity functions — so, unlike FLUX.1/FLUX.2/Qwen-Image, there is no sequence
packing to undo before decode. `mflux/models/krea2/variants/txt2img/krea2.py`'s `Krea2`
class declares `vae: QwenVAE` (imported from `mflux.models.qwen.model.qwen_vae.qwen_vae`),
confirming the VAE/weight-reuse premise. `mflux/models/common/config/model_config.py`'s
`AVAILABLE_MODELS["krea-2"]` entry carries `aliases=["krea-2", "krea2"]`.
"""

import mlx.core as mx

from mlx_taef.kernels._conversion import TaehvCombined
from mlx_taef.kernels._types import ArchSpec, LatentSpec, MfluxBinding, ModelKernel, UnpackContext
from mlx_taef.kernels.qwen import QWEN_IMAGE


def unpack_krea2_latent(latent: mx.array, ctx: UnpackContext) -> mx.array:
    """Unpack mflux's 4-D Krea 2 in-loop latent into NHWC (B, h, w, 16) for taew2.1.

    mflux hands the callback the plain NCHW latent: `Krea2LatentCreator.create_noise`
    produces (1, 16, height//8, width//8) and its pack_latents/unpack_latents are identity
    functions, so — unlike FLUX/Qwen — there is no sequence packing to undo. Only an
    NCHW->NHWC transpose is needed. `ctx.latent_height/width` are unused (the latent
    carries its spatial dims) but kept for the uniform `(latent, UnpackContext)` signature.
    """
    if latent.ndim != 4 or latent.shape[1] != 16:
        raise ValueError(f"Expected Krea 2 in-loop latent (B, 16, h, w), got {tuple(latent.shape)}")
    return mx.transpose(latent, (0, 2, 3, 1))


KREA2 = ModelKernel(
    name="krea2",
    arch=ArchSpec(name="taehv"),
    conversion=TaehvCombined(),
    latent=LatentSpec(channels=16),
    source=QWEN_IMAGE.source,  # shared converted-cache entry, like ZIMAGE reuses TAEF1.source
    integration=MfluxBinding(
        mflux_models=(
            "krea-2",
            "krea2",
        ),  # verified aliases, mflux 0.18.1 AVAILABLE_MODELS["krea-2"]
        unpack=unpack_krea2_latent,
        packed_latent_downscale=None,  # 4-D unpacked latent, like zimage — not packed like qwen
    ),
    memory_cap_hint_gb=1,
)
