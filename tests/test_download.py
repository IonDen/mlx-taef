"""Offline tests for download.get_or_convert: caching + role dispatch, no network."""

import stat

import mlx.core as mx
import pytest

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
    cache_mode = stat.S_IMODE((tmp_path / "converted").stat().st_mode)
    assert cache_mode == 0o700


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


@pytest.mark.network
@pytest.mark.parametrize("kernel_name", ["taesd", "taesdxl", "taef1", "taef2", "qwen-image"])
@pytest.mark.parametrize("role", ["decoder", "encoder"])
def test_pinned_runtime_source_downloads_verifies_and_converts(
    kernel_name: str, role: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Live trust-chain gate: immutable download -> role digest -> conversion -> MLX cache."""
    monkeypatch.setattr(download, "CACHE_ROOT", tmp_path / kernel_name)

    path = download.get_or_convert(get_kernel(kernel_name), role=role)

    assert path.exists()
    assert mx.load(str(path))
