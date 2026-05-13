"""Tests for HF Hub auto-download + cache."""

from pathlib import Path

import mlx.core as mx
import pytest


def test_get_or_convert_returns_cached_path_when_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call should return cached path without re-running conversion."""
    from mlx_taef import download
    from mlx_taef.variants import TAEF2_CONFIG

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    fake_call_count = [0]

    def fake_converter(*, out_path: Path, config) -> None:
        fake_call_count[0] += 1
        out_path.write_bytes(b"fake")

    monkeypatch.setattr(download, "convert_hf_decoder_to_mlx", fake_converter)

    p1 = download.get_or_convert(TAEF2_CONFIG, role="decoder")
    p2 = download.get_or_convert(TAEF2_CONFIG, role="decoder")

    assert p1 == p2
    assert fake_call_count[0] == 1, "Converter should be called exactly once"


def test_get_or_convert_dispatches_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_taef import download
    from mlx_taef.variants import TAEF2_CONFIG

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    calls = []

    def fake_dec(*, out_path: Path, config) -> None:
        calls.append(("decoder", out_path.name))
        out_path.write_bytes(b"fake")

    def fake_enc(*, out_path: Path, config) -> None:
        calls.append(("encoder", out_path.name))
        out_path.write_bytes(b"fake")

    monkeypatch.setattr(download, "convert_hf_decoder_to_mlx", fake_dec)
    monkeypatch.setattr(download, "convert_hf_encoder_to_mlx", fake_enc)

    download.get_or_convert(TAEF2_CONFIG, role="decoder")
    download.get_or_convert(TAEF2_CONFIG, role="encoder")

    assert calls == [
        ("decoder", "taef2_decoder.safetensors"),
        ("encoder", "taef2_encoder.safetensors"),
    ]


def test_get_or_convert_rejects_invalid_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlx_taef import download
    from mlx_taef.variants import TAEF2_CONFIG

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="role must be"):
        download.get_or_convert(TAEF2_CONFIG, role="invalid")


def test_from_pretrained_repo_id_mismatch_raises() -> None:
    from mlx_taef import TAEF2

    with pytest.raises(ValueError, match="repo_id mismatch"):
        TAEF2.from_pretrained(repo_id="some/other-repo")


@pytest.mark.network
def test_from_pretrained_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: full from_pretrained() against real HF download. Marked network."""
    from mlx_taef import TAEF2, download

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    model = TAEF2.from_pretrained(include_encoder=False)
    assert model is not None
    # Sanity: weights are actually loaded.
    latent = mx.zeros((1, 8, 8, 32))
    img = model.decode(latent)
    assert img.shape == (1, 64, 64, 3)
