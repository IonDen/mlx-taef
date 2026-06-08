"""Guards: decode/encode must raise (not run random-init modules) when weights are unloaded."""

import mlx.core as mx
import pytest

from mlx_taef import TAEF2, TaefError


def test_decode_on_unloaded_instance_raises_taef_error() -> None:
    model = TAEF2()  # built at random init, no weights loaded
    with pytest.raises(TaefError, match="from_pretrained"):
        model.decode(mx.zeros((1, 8, 8, 32)))


def test_decode_image_on_unloaded_instance_raises_taef_error() -> None:
    model = TAEF2()
    with pytest.raises(TaefError):
        model.decode_image(mx.zeros((1, 8, 8, 32)))


def test_encode_on_decoder_only_instance_raises_taef_error(converted_dir) -> None:
    model = TAEF2.from_pretrained_local(converted_dir / "taef2_decoder.safetensors")
    with pytest.raises(TaefError, match="include_encoder"):
        model.encode(mx.zeros((1, 64, 64, 3)))
