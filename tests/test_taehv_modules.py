"""Shape + param-layout tests for the taehv decoder/encoder modules (synthetic weights).

These check structure, the single-image (T=1) collapse, and that param keys follow the MLX
nn.Sequential layout (`layers.N...`) so the conversion can reuse `_build_mlx_state_dict`.
Bit-exact parity vs upstream is a later task (needs the real checkpoint).
"""

import mlx.core as mx

from mlx_taef.convert import _flatten_module_param_shapes
from mlx_taef.kernels._taehv import TaehvDecoder, TaehvEncoder


def test_decoder_t1_returns_single_8x_upscaled_frame():
    dec = TaehvDecoder(latent_channels=16)
    mx.eval(dec.parameters())
    out = dec(mx.zeros((1, 8, 8, 16)))  # B,H,W,16 ; H,W>1
    assert out.shape == (1, 64, 64, 3)  # 8x spatial, one RGB frame, channels last
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_encoder_t1_returns_single_latent_frame():
    enc = TaehvEncoder(latent_channels=16)
    mx.eval(enc.parameters())
    out = enc(mx.zeros((1, 64, 64, 3)))  # B,H,W,3 ; /8-divisible
    assert out.shape == (1, 8, 8, 16)  # /8 spatial, 16 channels, one latent frame


def test_decoder_param_keys_follow_sequential_layout():
    # The conversion strategy (Task 7) relies on these matching _sequential_key_to_mlx output.
    keys = _flatten_module_param_shapes(TaehvDecoder(latent_channels=16))
    assert keys["layers.1.weight"] == (256, 3, 3, 16)  # conv(16->256), NHWC (out,kH,kW,in)
    assert "layers.3.conv.layers.0.weight" in keys  # MemBlock(256,256) first conv
    assert "layers.7.conv.weight" in keys  # TGrow(256,1) 1x1 conv
    assert keys["layers.22.weight"] == (3, 3, 3, 64)  # final conv(64->3)


def test_encoder_param_keys_follow_sequential_layout():
    keys = _flatten_module_param_shapes(TaehvEncoder(latent_channels=16))
    assert keys["layers.0.weight"] == (64, 3, 3, 3)  # conv(3->64)
    assert "layers.2.conv.weight" in keys  # TPool(64,2) 1x1 conv
    assert "layers.4.conv.layers.0.weight" in keys  # MemBlock first conv
    assert keys["layers.17.weight"] == (16, 3, 3, 64)  # conv(64->16)
