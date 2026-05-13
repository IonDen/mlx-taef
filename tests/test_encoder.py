"""Tier 2 parity tests for encoder side of all variants."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_taef import TAEF1, TAEF2, TAESD, TAESDXL, Taef
from mlx_taef.variants import (
    TAEF1_CONFIG,
    TAEF2_CONFIG,
    TAESD_CONFIG,
    TAESDXL_CONFIG,
    TaesdVariantConfig,
)

CONVERTED_DIR = Path(__file__).parent / "converted"
REF_DIR = Path(__file__).parent / "reference"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _load_source_image() -> mx.array:
    src = (
        np.array(Image.open(REF_DIR / "_source_image.png").convert("RGB")).astype(np.float32)
        / 255.0
    )
    return mx.array(src[None, ...])  # NHWC (1, 256, 256, 3)


@pytest.mark.parametrize(
    ("variant_name", "cls", "config"),
    [
        ("taesd", TAESD, TAESD_CONFIG),
        ("taesdxl", TAESDXL, TAESDXL_CONFIG),
        ("taef1", TAEF1, TAEF1_CONFIG),
        ("taef2", TAEF2, TAEF2_CONFIG),
    ],
)
def test_variant_encode_parity(
    variant_name: str,
    cls: type[Taef],
    config: TaesdVariantConfig,
) -> None:
    """Encoded latents must match PyTorch reference at cosine similarity > 0.999."""
    model = cls.from_pretrained_local(
        decoder_path=CONVERTED_DIR / f"{variant_name}_decoder.safetensors",
        encoder_path=CONVERTED_DIR / f"{variant_name}_encoder.safetensors",
    )
    src = _load_source_image()
    expected = np.array(mx.load(str(REF_DIR / f"{variant_name}_encoded_001.safetensors"))["latent"])
    actual = np.array(model.encode(src))
    sim = _cosine_sim(actual, expected)
    assert sim > 0.999, f"{variant_name}: encode cos_sim={sim:.6f}"


def test_taef2_encode_shape_matches_8x_downsample() -> None:
    """Encoder downsamples by 8x spatially. Source 256x256 -> 32x32 latent."""
    model = TAEF2.from_pretrained_local(
        decoder_path=CONVERTED_DIR / "taef2_decoder.safetensors",
        encoder_path=CONVERTED_DIR / "taef2_encoder.safetensors",
    )
    src = mx.zeros((1, 256, 256, 3))
    latent = model.encode(src)
    # TAEF2 has latent_channels=32
    assert latent.shape == (1, 32, 32, 32), f"Got {latent.shape}"
