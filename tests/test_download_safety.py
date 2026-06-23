"""cache_key path-safety + the source sha256 integrity check."""

import hashlib
from pathlib import Path

import pytest

from mlx_taef.errors import ConversionError
from mlx_taef.kernels._conversion import _verify_sha256
from mlx_taef.kernels._types import WeightSource


def test_cache_key_sanitizes_slash_and_dotdot():
    # The canonical filename can contain a subdir ("safetensors/..."); cache_key must not let
    # that escape the cache directory.
    src = WeightSource(repo="IonDen/taew2.1", filename="safetensors/../taew2_1.safetensors")
    key = src.cache_key(role="decoder")
    assert "/" not in key
    assert ".." not in key


def test_verify_sha256_noop_when_expected_is_none(tmp_path: Path):
    f = tmp_path / "w"
    f.write_bytes(b"abc")
    _verify_sha256(f, None)  # must not raise


def test_verify_sha256_passes_on_match(tmp_path: Path):
    f = tmp_path / "w"
    f.write_bytes(b"abc")
    _verify_sha256(f, hashlib.sha256(b"abc").hexdigest())  # must not raise


def test_verify_sha256_raises_on_mismatch(tmp_path: Path):
    f = tmp_path / "w"
    f.write_bytes(b"abc")
    with pytest.raises(ConversionError):
        _verify_sha256(f, "0" * 64)
