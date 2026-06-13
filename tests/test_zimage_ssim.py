"""Z-Image SSIM gate (network). Proves TAEF1-weight reuse holds vs mflux's full Z-Image VAE."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "showcase_latents" / "z_image_turbo.safetensors"


@pytest.mark.network
def test_zimage_taef1_decode_matches_full_vae_ssim() -> None:
    """SSIM(TAEF1 preview, full Z-Image VAE) >= 0.75 on the committed (red apple, seed 42) latent."""
    from skimage.metrics import structural_similarity as ssim

    from mlx_taef import ZImage
    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.zimage import unpack_zimage_latent

    blob = mx.load(str(FIXTURE))
    latent = blob["latent"]  # (16, 1, h, w)
    height = int(np.array(blob["height"])[0])
    width = int(np.array(blob["width"])[0])
    assert latent.shape == (16, 1, height // 8, width // 8), latent.shape

    # Path A — mlx-taef Z-Image preview (TAEF1 weights), offline.
    taef = ZImage.from_pretrained(include_encoder=False)
    ctx = UnpackContext(latent_height=height // 8, latent_width=width // 8)
    preview = np.array(taef.decode_image(unpack_zimage_latent(latent, ctx))[0])  # (H,W,3) uint8

    # Path B — mflux full Z-Image VAE (network).
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
    from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

    mflux_model = MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo())
    unpacked = ZImageLatentCreator.unpack_latents(latent, height, width)  # (1,16,h,w)
    full = VAEUtil.decode(vae=mflux_model.vae, latent=unpacked, tiling_config=None)
    mx.eval(full)
    # mflux decode is NCHW in [-1,1]; match ImageUtil._denormalize (x/2+0.5) then to HWC uint8.
    full_uint8 = np.array((mx.clip(full / 2 + 0.5, 0, 1) * 255).astype(mx.uint8))
    full_uint8 = np.transpose(full_uint8[0], (1, 2, 0))  # (H,W,3)

    assert preview.ndim == 3
    assert preview.shape[-1] == 3
    assert full_uint8.ndim == 3
    assert full_uint8.shape[-1] == 3
    assert preview.shape == full_uint8.shape
    assert full_uint8.min() < 50  # sanity: not an all-grey degenerate decode

    score = ssim(preview, full_uint8, channel_axis=-1, data_range=255)
    print(f"\nZ-Image TAEF1-vs-full-VAE SSIM = {score:.4f}")
    assert score >= 0.75, f"Z-Image TAEF1 reuse falsified: SSIM {score:.3f} < 0.75"
