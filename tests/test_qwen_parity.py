"""Bit-exact parity: MLX qwen-image (taew2.1) decode/encode vs the committed upstream oracle.

Offline (committed reference fixtures + committed converted MLX weights). Measured worst maxabs
on an M1 Max (MLX-Metal fp32 vs PyTorch fp32): decode ~2.9e-6, encode ~3.1e-6 — far under the
atols below, which reuse the established per-type cross-hardware bounds (decode 1e-4, encode 1e-3;
see test_api.py / test_encoder.py).
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_taef import QwenImage

REF = Path(__file__).parent / "reference"
CONV = Path(__file__).parent / "converted"
DECODE_ATOL = 1e-4
ENCODE_ATOL = 1e-3


@pytest.mark.parametrize("i", range(5))
def test_qwen_decode_matches_reference(i: int) -> None:
    latent = mx.load(str(REF / f"qwen-image_latent_{i:03d}.safetensors"))["latent"]
    ref = np.asarray(mx.load(str(REF / f"qwen-image_decoded_{i:03d}.safetensors"))["image"])
    model = QwenImage.from_pretrained_local(CONV / "qwen-image_decoder.safetensors")
    out = np.asarray(model.decode(latent))
    assert out.shape == ref.shape == (1, 128, 128, 3)
    np.testing.assert_allclose(out, ref, atol=DECODE_ATOL, rtol=0)


def test_qwen_encode_matches_reference() -> None:
    src = Image.open(REF / "_source_image.png").convert("RGB").resize((256, 256))
    img = mx.array((np.array(src).astype(np.float32) / 255.0)[None])
    ref = np.asarray(mx.load(str(REF / "qwen-image_encoded_001.safetensors"))["latent"])
    model = QwenImage.from_pretrained_local(
        CONV / "qwen-image_decoder.safetensors",
        encoder_path=CONV / "qwen-image_encoder.safetensors",
    )
    out = np.asarray(model.encode(img))
    assert out.shape == ref.shape == (1, 32, 32, 16)
    np.testing.assert_allclose(out, ref, atol=ENCODE_ATOL, rtol=0)
