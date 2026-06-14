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


def test_decode_wrong_channel_count_raises_value_error(converted_dir) -> None:
    model = TAEF2.from_pretrained_local(converted_dir / "taef2_decoder.safetensors")
    # TAEF2 expects 32 latent channels; feed a 16-channel (FLUX.1) latent.
    with pytest.raises(ValueError, match="taef2"):
        model.decode(mx.zeros((1, 8, 8, 16)))


def test_encode_non_rgb_raises_value_error(converted_dir) -> None:
    model = TAEF2.from_pretrained_local(
        decoder_path=converted_dir / "taef2_decoder.safetensors",
        encoder_path=converted_dir / "taef2_encoder.safetensors",
    )
    with pytest.raises(ValueError, match="3 channels"):
        model.encode(mx.zeros((1, 64, 64, 2)))


# Wrong-rank inputs whose last dim still matches the expected channel count slip past the
# channel-count guard and would otherwise fail deep in the conv stack with an opaque error.
@pytest.mark.parametrize("bad_shape", [(8, 8, 32), (32,)])
def test_decode_wrong_rank_raises_value_error(converted_dir, bad_shape) -> None:
    model = TAEF2.from_pretrained_local(converted_dir / "taef2_decoder.safetensors")
    with pytest.raises(ValueError, match="4-D"):
        model.decode(mx.zeros(bad_shape))


@pytest.mark.parametrize("bad_shape", [(64, 64, 3), (3,)])
def test_encode_wrong_rank_raises_value_error(converted_dir, bad_shape) -> None:
    model = TAEF2.from_pretrained_local(
        decoder_path=converted_dir / "taef2_decoder.safetensors",
        encoder_path=converted_dir / "taef2_encoder.safetensors",
    )
    with pytest.raises(ValueError, match="4-D"):
        model.encode(mx.zeros(bad_shape))
