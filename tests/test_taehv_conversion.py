"""Conversion of the combined taew2.1 checkpoint covers the full taehv arch (decoder + encoder).

Network-marked: downloads the canonical (sha256-verified) checkpoint. Bit-exact VALUE parity vs
the upstream oracle is a separate fixture test; this asserts the remap produces every expected
parameter with the right shape + dtype (the strict coverage-verify raises otherwise).
"""

import mlx.core as mx
import pytest
from safetensors.numpy import load_file

from mlx_taef.convert import _build_mlx_state_dict, _flatten_module_param_shapes
from mlx_taef.kernels._arch import build_arch
from mlx_taef.kernels._conversion import TaehvCombined
from tests._taehv_canonical import canonical_taew21_path


@pytest.mark.network
@pytest.mark.parametrize("role", ["decoder", "encoder"])
def test_taehv_conversion_covers_full_arch(role: str) -> None:
    full = load_file(str(canonical_taew21_path()))
    raw = TaehvCombined._select_role(full, role)
    arch = build_arch("taehv", role=role, latent_channels=16, midblock_gn=False)
    expected = _flatten_module_param_shapes(arch)
    converted = _build_mlx_state_dict(
        raw, expected_shapes=expected
    )  # raises ConversionError if incomplete
    assert set(converted) == set(expected)
    for key, shape in expected.items():
        assert tuple(converted[key].shape) == tuple(shape)
        assert converted[key].dtype == mx.float32


def test_select_role_strips_prefix_and_casts_fp32() -> None:
    import numpy as np

    full = {
        "decoder.1.weight": np.zeros((4, 4), dtype=np.float16),
        "encoder.0.weight": np.zeros((2, 2), dtype=np.float16),
    }
    dec = TaehvCombined._select_role(full, "decoder")
    assert set(dec) == {"1.weight"}
    assert dec["1.weight"].dtype == np.float32
