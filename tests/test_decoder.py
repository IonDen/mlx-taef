"""Tests for the decoder factory."""

import mlx.core as mx

from mlx_taef.model import make_decoder
from mlx_taef.variants import TAEF1_CONFIG, TAEF2_CONFIG, TAESD_CONFIG


def test_taef2_decoder_output_shape():
    decoder = make_decoder(TAEF2_CONFIG)
    # Input: NHWC latent (1, 64, 64, 32) -> output (1, 512, 512, 3) at 8x upsample
    latent = mx.zeros((1, 64, 64, 32))
    out = decoder(latent)
    assert out.shape == (1, 512, 512, 3)


def test_taef1_decoder_output_shape():
    decoder = make_decoder(TAEF1_CONFIG)
    latent = mx.zeros((1, 64, 64, 16))
    out = decoder(latent)
    assert out.shape == (1, 512, 512, 3)


def test_taesd_decoder_output_shape():
    decoder = make_decoder(TAESD_CONFIG)
    latent = mx.zeros((1, 64, 64, 4))
    out = decoder(latent)
    assert out.shape == (1, 512, 512, 3)


def test_decoder_module_count_matches_upstream():
    """Upstream has: Clamp + Conv + ReLU + (Block x3 + Up + Conv) x3 + Block + Conv = 20 layers."""
    decoder = make_decoder(TAEF2_CONFIG)
    assert len(decoder.layers) == 20


def test_taef2_decoder_uses_midblock_gn_only_in_first_three_blocks():
    """In flux_2 arch, ONLY blocks 3, 4, 5 (after Clamp, Conv, ReLU at indices 0, 1, 2) have midblock_gn."""
    decoder = make_decoder(TAEF2_CONFIG)
    layers = decoder.layers
    # Index 3, 4, 5: midblock_gn Blocks
    for i in [3, 4, 5]:
        assert layers[i].pool is not None, f"Layer {i} should have midblock_gn pool"
    # Remaining Blocks at indices 8, 9, 10, 13, 14, 15, 18 should NOT have midblock_gn
    for i in [8, 9, 10, 13, 14, 15, 18]:
        assert layers[i].pool is None, f"Layer {i} should NOT have midblock_gn pool"


def test_taesd_decoder_has_no_midblock_gn():
    """Standard (non-flux_2) arch has no midblock_gn in any Block."""
    decoder = make_decoder(TAESD_CONFIG)
    blocks = [layer for layer in decoder.layers if hasattr(layer, "pool")]
    assert len(blocks) == 10, f"expected 10 Block layers, found {len(blocks)}"
    assert all(b.pool is None for b in blocks), "no TAESD Block should have a midblock_gn pool"
