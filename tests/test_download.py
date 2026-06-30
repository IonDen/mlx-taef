"""Offline tests for download.get_or_convert: caching + role dispatch, no network."""

import mlx.core as mx

from mlx_taef import download
from mlx_taef.kernels import get_kernel


def _fake_convert_factory(calls):
    def _fake(self, source, arch_module, *, role):
        # arch_module must be the built arch (a real nn module), not None — a build/wiring
        # regression that dropped it would otherwise pass unnoticed.
        assert arch_module is not None
        assert hasattr(arch_module, "parameters")
        calls.append(role)
        return {"w": mx.zeros((1,))}

    return _fake


def test_get_or_convert_caches_and_dispatches_role(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        "mlx_taef.kernels._conversion.DiffusersRemap.convert", _fake_convert_factory(calls)
    )
    k = get_kernel("taef1")

    p_dec = download.get_or_convert(k, role="decoder")
    assert p_dec.exists()
    assert calls == ["decoder"]

    calls.clear()
    p_dec2 = download.get_or_convert(k, role="decoder")
    assert p_dec2 == p_dec
    assert calls == []

    p_enc = download.get_or_convert(k, role="encoder")
    assert p_enc != p_dec
    assert calls == ["encoder"]


def test_get_or_convert_rejects_bad_role(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="role must be"):
        download.get_or_convert(get_kernel("taef1"), role="bogus")


def test_get_or_convert_leaves_no_partial_cache_on_interrupted_write(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        "mlx_taef.kernels._conversion.DiffusersRemap.convert",
        lambda self, source, arch_module, *, role: {"w": mx.zeros((1,))},
    )

    def _failing_save(path, arrays):
        # Simulate a process killed mid-write: bytes land on disk, then it dies.
        from pathlib import Path

        Path(path).write_bytes(b"truncated-safetensors-header")
        raise RuntimeError("disk full mid-write")

    monkeypatch.setattr(download.mx, "save_safetensors", _failing_save)

    k = get_kernel("taef1")
    with pytest.raises(RuntimeError, match="disk full mid-write"):
        download.get_or_convert(k, role="decoder")

    cache_dir = tmp_path / "converted"
    out_path = cache_dir / f"{k.source.cache_key(role='decoder')}.mlx.safetensors"
    assert not out_path.exists(), "an interrupted write must not leave a usable-looking cache file"
    assert list(cache_dir.glob("*")) == [], "an interrupted write must not leave stray temp files"
