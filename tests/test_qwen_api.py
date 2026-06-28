"""QwenImage is exported, bound to the qwen-image kernel, and kept out of the legacy convert CLI."""


def test_qwenimage_exported_and_bound():
    import mlx_taef
    from mlx_taef import QwenImage

    assert "QwenImage" in mlx_taef.__all__
    assert QwenImage._kernel.name == "qwen-image"


def test_qwenimage_excluded_from_legacy_variants():
    # The legacy `convert` CLI path is TAESD-only; qwen-image (like zimage) is excluded.
    from mlx_taef.variants import VARIANTS

    assert "qwen-image" not in VARIANTS


def test_qwenimage_in_bench_cls_map():
    from mlx_taef.cli import _BENCH_CLS_BY_NAME

    assert "qwen-image" in _BENCH_CLS_BY_NAME


def test_qwenimage_constructs_and_decodes_qwen_shaped_latent():
    """API-level behavioral smoke: construct QwenImage from committed weights and decode a
    16-channel Wan 2.1 latent. A broken taehv build/decode reds this (the tests above only
    check exports/registry membership, never construction or the forward pass)."""
    from pathlib import Path

    import mlx.core as mx

    from mlx_taef import QwenImage

    conv = Path(__file__).parent / "converted" / "qwen-image_decoder.safetensors"
    model = QwenImage.from_pretrained_local(conv)
    assert model._kernel.latent.channels == 16
    out = model.decode(mx.zeros((1, 16, 16, 16)))  # taew2.1 decodes the 16-ch latent at 8x upscale
    mx.eval(out)
    assert out.shape == (1, 128, 128, 3)
