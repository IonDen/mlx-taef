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
