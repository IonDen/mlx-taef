"""Bit-exact parity: MLX qwen-image (taew2.1) decode/encode vs the committed upstream oracle.

Offline (committed reference fixtures + committed converted MLX weights). Both the oracle
(PyTorch) and the port (MLX-Metal) run fp32, so the gate is tight: measured worst maxabs on an
M1 Max is ~2.9e-6 (decode) / ~3.1e-6 (encode), and the 1e-5 atol is ~3x that — sensitive enough
to catch a real accumulation/transpose regression, unlike the looser fp16-grade bounds the
cross-hardware TAESD variants use. If a future runner exceeds 1e-5, loosen with a documented
cross-hardware measurement rather than reverting to an fp16 tolerance.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_taef import QwenImage

REF = Path(__file__).parent / "reference"
CONV = Path(__file__).parent / "converted"
DECODE_ATOL = 1e-5
ENCODE_ATOL = 1e-5


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


def test_qwen_kernel_binding_unpacks_then_decodes() -> None:
    # Drive the kernel's mflux binding end-to-end (binding.unpack -> decode) so a broken or
    # wrong-shape unpack wiring is caught offline, not only at live-preview time.
    from mlx_taef.kernels import get_kernel
    from mlx_taef.kernels._types import UnpackContext

    binding = get_kernel("qwen-image").integration
    assert binding is not None
    lh, lw = 8, 8
    packed = mx.random.normal((1, lh * lw, 64), key=mx.random.key(1))
    latent = binding.unpack(packed, UnpackContext(latent_height=lh, latent_width=lw))
    assert latent.shape == (1, lh * 2, lw * 2, 16)
    model = QwenImage.from_pretrained_local(CONV / "qwen-image_decoder.safetensors")
    out = model.decode_image(latent)
    mx.eval(out)
    assert out.shape == (1, lh * 2 * 8, lw * 2 * 8, 3)  # decode upscales 8x spatial
    assert bool(mx.all(out <= 255).item())
