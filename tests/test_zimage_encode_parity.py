"""Z-Image encode-parity gate (network). Cross-roundtrip vs mflux's full Z-Image VAE.

`ZImage` inherits TAEF1's encoder (only the decode direction is SSIM-gated in
`test_zimage_ssim.py`). This test measures whether that reused encoder holds up on the
opposite direction: real image -> `ZImage.encode()` -> mflux's full Z-Image VAE decode ->
SSIM against the original image.

Needs no new committed asset: the "real photographic image" is obtained by decoding the
already-committed `tests/fixtures/showcase_latents/z_image_turbo.safetensors` latent with
mflux's full Z-Image VAE (the same latent `test_zimage_ssim.py` uses for its decode-direction
gate).

Threshold: `assert score >= 0.75` (measured 0.9580 on M1 Max, mflux 0.18.1, quantize=4,
2026-08-09).
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "showcase_latents" / "z_image_turbo.safetensors"


@pytest.mark.network
def test_zimage_encode_roundtrip_matches_input_ssim() -> None:
    """SSIM(input image, encode->full-VAE-decode roundtrip) >= 0.75.

    Cross-roundtrip: a real image (decoded from the committed fixture latent via mflux's full
    Z-Image VAE) is re-encoded with `ZImage`'s TAEF1 encoder, then decoded back with mflux's
    full VAE. The threshold (measured 0.9580 on M1 Max, mflux 0.18.1, quantize=4, 2026-08-09)
    is set on the *encode* direction specifically, mirroring `test_zimage_ssim.py`'s decode
    gate.
    """
    from skimage.metrics import structural_similarity as ssim

    from mlx_taef import ZImage

    blob = mx.load(str(FIXTURE))
    latent = blob["latent"]  # (16, 1, h, w)
    height = int(np.array(blob["height"])[0])
    width = int(np.array(blob["width"])[0])
    assert latent.shape == (16, 1, height // 8, width // 8), latent.shape

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
    from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

    mflux_model = MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo())

    # Step 1: build the "real" input image by decoding the committed fixture latent with
    # mflux's full Z-Image VAE. mflux's unpack turns the committed (16,1,h,w) in-loop latent
    # into the (1,16,h,w) NCHW layout VAEUtil.decode expects.
    unpacked = ZImageLatentCreator.unpack_latents(latent, height, width)  # (1,16,h,w) NCHW
    input_full = VAEUtil.decode(vae=mflux_model.vae, latent=unpacked, tiling_config=None)
    mx.eval(input_full)  # NCHW, [-1,1] (mflux VAE decode convention)

    # NCHW [-1,1] -> NHWC [0,1]: ZImage.encode() (like ImageUtil._denormalize) expects
    # channel-last RGB in [0,1] float.
    input_01_nhwc = mx.clip(input_full / 2 + 0.5, 0.0, 1.0).transpose(0, 2, 3, 1)
    mx.eval(input_01_nhwc)
    input_uint8 = np.array((input_01_nhwc * 255.0).astype(mx.uint8))[0]  # NHWC[0] -> (H,W,3)

    # Sanity: not an all-grey degenerate decode (mirrors test_zimage_ssim.py's check).
    assert input_uint8.min() < 50

    # Step 2: encode the real image with ZImage's (TAEF1) encoder. encode() takes NHWC [0,1]
    # and returns a raw latent in NHWC (1, h, w, 16) -- the same raw-latent scale `decode()`
    # consumes directly (no extra scale_latents() step; see api.py's decode()/encode()).
    taef = ZImage.from_pretrained(include_encoder=True)
    roundtrip_latent_nhwc = taef.encode(input_01_nhwc)  # (1, h, w, 16), raw latent scale
    mx.eval(roundtrip_latent_nhwc)

    # NHWC -> NCHW: VAEUtil.decode expects channel-first, matching the raw latent layout
    # ZImageLatentCreator.unpack_latents produces above.
    roundtrip_latent_nchw = roundtrip_latent_nhwc.transpose(0, 3, 1, 2)  # (1, 16, h, w)

    # Step 3: decode the re-encoded latent with mflux's full Z-Image VAE (same decoder used
    # to build the input image, so the comparison isolates the encode direction).
    roundtrip_full = VAEUtil.decode(
        vae=mflux_model.vae, latent=roundtrip_latent_nchw, tiling_config=None
    )
    mx.eval(roundtrip_full)  # NCHW, [-1,1]

    # NCHW [-1,1] -> HWC uint8 for the roundtrip image: SSIM wants uint8 arrays with an
    # explicit channel axis, same denormalize convention used for input_uint8 above.
    roundtrip_uint8 = np.array(
        (mx.clip(roundtrip_full / 2 + 0.5, 0.0, 1.0) * 255.0).astype(mx.uint8)
    )
    roundtrip_uint8 = np.transpose(roundtrip_uint8[0], (1, 2, 0))  # (H,W,3)

    assert input_uint8.ndim == 3
    assert input_uint8.shape[-1] == 3
    assert roundtrip_uint8.ndim == 3
    assert roundtrip_uint8.shape[-1] == 3
    assert input_uint8.shape == roundtrip_uint8.shape

    score = ssim(input_uint8, roundtrip_uint8, channel_axis=-1, data_range=255)
    print(f"\nZ-Image encode-roundtrip SSIM = {score:.4f}")
    assert score >= 0.75, f"Z-Image encode reuse falsified: SSIM {score:.3f} < 0.75"
