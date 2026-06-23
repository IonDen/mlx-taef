"""Perceptual + dtype sanity for qwen-image (offline, committed weights).

The strong correctness guarantee is the bit-exact parity vs the upstream oracle
(test_qwen_parity). These add a perceptual floor and an fp16 smoke check.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from mlx_taef import QwenImage

REF = Path(__file__).parent / "reference"
CONV = Path(__file__).parent / "converted"


def test_qwen_roundtrip_ssim_floor():
    # taew2.1 encode->decode self-roundtrip on the shared source image. Measured ~0.68 on an
    # M1 Max — this is taew2.1's inherent tiny-AE roundtrip quality (the port is bit-exact to
    # upstream; see test_qwen_parity). The live-preview quality vs the full Wan VAE needs a real
    # Qwen-Image latent (the ~20B model) and is community-measured. This floor guards gross breakage.
    model = QwenImage.from_pretrained_local(
        CONV / "qwen-image_decoder.safetensors",
        encoder_path=CONV / "qwen-image_encoder.safetensors",
    )
    src = (
        np.array(Image.open(REF / "_source_image.png").convert("RGB").resize((256, 256))).astype(
            np.float32
        )
        / 255.0
    )
    recon = np.asarray(model.decode(model.encode(mx.array(src[None]))))[0]
    assert ssim(src, recon, channel_axis=-1, data_range=1.0) >= 0.65


def test_qwen_decode_fp16_runs_and_is_finite():
    # The bench path runs fp16; confirm the recurrent module tolerates set_dtype + decode.
    model = QwenImage.from_pretrained_local(
        CONV / "qwen-image_decoder.safetensors", dtype=mx.float16
    )
    out = model.decode(mx.zeros((1, 8, 8, 16), dtype=mx.float16))
    mx.eval(out)
    assert out.shape == (1, 64, 64, 3)
    assert bool(mx.all(mx.isfinite(out)).item())
