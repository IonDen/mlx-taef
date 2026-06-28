import numpy as np

from mlx_taef.convert import _flatten_module_param_shapes
from mlx_taef.kernels._arch import build_arch
from mlx_taef.kernels._conversion import DiffusersRemap, TaehvCombined, UpstreamTwoFile


def _mlx_to_sequential(mlx_key: str) -> str:
    # Inverse of convert._sequential_key_to_mlx: drop the "layers" path segments.
    return ".".join(p for p in mlx_key.split(".") if p != "layers")


def _synth_sequential_source(arch_module) -> dict[str, np.ndarray]:
    # 4D conv weights stored NCHW (out,in,kh,kw) so the strategy's NHWC transpose runs.
    expected = _flatten_module_param_shapes(arch_module)
    src: dict[str, np.ndarray] = {}
    for mlx_key, shape in expected.items():
        seq_key = _mlx_to_sequential(mlx_key)
        if len(shape) == 4:
            o, kh, kw, i = shape
            src[seq_key] = np.zeros((o, i, kh, kw), dtype=np.float32)
        else:
            src[seq_key] = np.zeros(shape, dtype=np.float32)
    return src


def test_diffusers_remap_round_trips_to_arch_shapes_offline(monkeypatch):
    decoder = build_arch("taesd2d", role="decoder", latent_channels=16, midblock_gn=False)
    expected = _flatten_module_param_shapes(decoder)
    src = _synth_sequential_source(decoder)
    strat = DiffusersRemap()
    monkeypatch.setattr(strat, "_load_raw", lambda source, role: src)
    out = strat.convert(source=object(), arch_module=decoder, role="decoder")  # type: ignore[arg-type]
    assert set(out) == set(expected)
    for k, shape in expected.items():
        assert tuple(out[k].shape) == shape


def test_upstream_two_file_round_trips_offline(monkeypatch):
    encoder = build_arch("taesd2d", role="encoder", latent_channels=4, midblock_gn=False)
    expected = _flatten_module_param_shapes(encoder)
    src = _synth_sequential_source(encoder)
    strat = UpstreamTwoFile()
    monkeypatch.setattr(strat, "_load_raw", lambda source, role: src)
    out = strat.convert(source=object(), arch_module=encoder, role="encoder")  # type: ignore[arg-type]
    assert set(out) == set(expected)


def test_taehv_combined_round_trips_to_arch_shapes_offline(monkeypatch):
    """The qwen-image conversion path (taehv arch): convert() must produce arch-shaped MLX keys
    via the shared NCHW->NHWC transpose + coverage. Offline — _load_raw (download) is faked."""
    decoder = build_arch("taehv", role="decoder", latent_channels=16, midblock_gn=False)
    expected = _flatten_module_param_shapes(decoder)
    src = _synth_sequential_source(decoder)
    strat = TaehvCombined()
    monkeypatch.setattr(strat, "_load_raw", lambda source, role: src)
    out = strat.convert(source=object(), arch_module=decoder, role="decoder")  # type: ignore[arg-type]
    assert set(out) == set(expected)
    for k, shape in expected.items():
        assert tuple(out[k].shape) == shape
