"""Krea 2 SSIM gate (network). Proves taew2.1-weight reuse holds vs mflux's full Krea 2 VAE."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "showcase_latents" / "krea_2_turbo.safetensors"


@pytest.mark.network
def test_krea2_taew21_decode_matches_full_vae_ssim() -> None:
    """SSIM(taew2.1 preview, full Krea 2 VAE) >= 0.75 on the committed (red apple, seed 42) latent.

    Threshold is the family-wide floor, not the expectation: measured 0.9678 on M1 Max
    (mflux 0.18.1, quantize=4 full-VAE path, 2026-08-09).
    """
    from skimage.metrics import structural_similarity as ssim

    from mlx_taef import Krea2
    from mlx_taef.kernels import UnpackContext
    from mlx_taef.kernels.krea2 import unpack_krea2_latent

    blob = mx.load(str(FIXTURE))
    latent = blob["latent"]  # (1, 16, h, w)
    height = int(np.array(blob["height"])[0])
    width = int(np.array(blob["width"])[0])
    assert latent.shape == (1, 16, height // 8, width // 8), latent.shape

    # Path A — mlx-taef Krea 2 preview (taew2.1 weights, shared with the qwen-image kernel),
    # offline.
    taef = Krea2.from_pretrained(include_encoder=False)
    ctx = UnpackContext(latent_height=height // 8, latent_width=width // 8)
    preview = np.array(taef.decode_image(unpack_krea2_latent(latent, ctx))[0])  # (H,W,3) uint8

    # Path B — mflux full Krea 2 VAE (network). Krea2LatentCreator.unpack_latents is identity
    # (mflux/models/krea2/latent_creator/krea2_latent_creator.py; confirmed against installed
    # mflux 0.18.1, see task-2-report.md) — the committed fixture is already the raw 4-D NCHW
    # latent the VAE expects, so no unpack call is needed before decoding it directly.
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2 as MfluxKrea2

    mflux_model = MfluxKrea2(quantize=4, model_config=ModelConfig.krea2())
    full = VAEUtil.decode(vae=mflux_model.vae, latent=latent, tiling_config=None)
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
    print(f"\nKrea 2 taew2.1-vs-full-VAE SSIM = {score:.4f}")
    assert score >= 0.75, f"Krea 2 taew2.1 reuse falsified: SSIM {score:.3f} < 0.75"
