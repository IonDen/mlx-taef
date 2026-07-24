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
    src = WeightSource(repo="ionden/taew2.1", filename="safetensors/../taew2_1.safetensors")
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


def test_download_and_verify_pins_revision_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_download_and_verify must pass source.revision to hf_hub_download and return the
    downloaded path, so a revision pin on ANY source is honored (not just TaehvCombined)."""
    import hashlib

    import huggingface_hub

    from mlx_taef.kernels._conversion import _download_and_verify
    from mlx_taef.kernels._types import WeightSource

    payload = b"weights-bytes"
    f = tmp_path / "w.safetensors"
    f.write_bytes(payload)
    good_sha = hashlib.sha256(payload).hexdigest()

    captured: dict[str, object] = {}

    def _fake_download(*, repo_id: str, filename: str, revision: object = None) -> str:
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        return str(f)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    src = WeightSource(
        repo="x/y", filename="w.safetensors", revision="abc123def456", sha256=good_sha
    )
    out = _download_and_verify(src, "w.safetensors", role="decoder")
    assert out == f
    assert captured["revision"] == "abc123def456"
    assert captured["repo_id"] == "x/y"


def test_download_and_verify_raises_on_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    from mlx_taef.errors import ConversionError
    from mlx_taef.kernels._conversion import _download_and_verify
    from mlx_taef.kernels._types import WeightSource

    f = tmp_path / "w.safetensors"
    f.write_bytes(b"actual-bytes")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *, repo_id, filename, revision=None: str(f),
    )
    src = WeightSource(repo="x/y", filename="w.safetensors", sha256="0" * 64)
    with pytest.raises(ConversionError, match="sha256 mismatch"):
        _download_and_verify(src, "w.safetensors", role="decoder")


def test_pinned_sources_route_through_a_verifying_strategy() -> None:
    """Every sha256-pinned kernel must use one of the known verifying strategy classes. This
    guards against wiring a pinned source to a brand-new, unverified strategy type; the actual
    sha256-verification behavior of each strategy is covered by the per-strategy routing tests
    and test_download_and_verify_raises_on_sha_mismatch."""
    from mlx_taef.kernels import KERNELS
    from mlx_taef.kernels._conversion import (
        DiffusersRemap,
        TaehvCombined,
        UpstreamTwoFile,
    )

    verifying = (DiffusersRemap, UpstreamTwoFile, TaehvCombined)
    for name, kernel in KERNELS.items():
        if any(kernel.source.sha256_for(role) is not None for role in ("decoder", "encoder")):
            assert isinstance(kernel.conversion, verifying), (
                f"kernel {name!r} pins sha256 but its strategy does not verify it"
            )


def test_diffusers_remap_load_raw_routes_through_download_and_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DiffusersRemap._load_raw must fetch via _download_and_verify (pins revision + checks
    sha256), not hf_hub_download directly — otherwise a diffusers-source pin is ignored."""
    import safetensors.numpy

    import mlx_taef.convert as convert_mod
    import mlx_taef.kernels._conversion as conv
    from mlx_taef.kernels._conversion import DiffusersRemap
    from mlx_taef.kernels._types import WeightSource

    captured: dict[str, object] = {}

    def _fake_dav(source: object, filename: str, *, role: str) -> Path:
        captured["source"] = source
        captured["filename"] = filename
        captured["role"] = role
        return Path("/tmp/fake.safetensors")

    monkeypatch.setattr(conv, "_download_and_verify", _fake_dav)
    monkeypatch.setattr(safetensors.numpy, "load_file", lambda p: {"raw": 1})
    monkeypatch.setattr(
        convert_mod, "convert_diffusers_to_sequential", lambda sd, *, role: {"remapped": role}
    )

    src = WeightSource(repo="x/y", filename="w.safetensors")
    out = DiffusersRemap()._load_raw(src, "decoder")
    assert captured["filename"] == "w.safetensors"
    assert captured["source"] is src
    assert captured["role"] == "decoder"
    assert out == {"remapped": "decoder"}


def test_upstream_two_file_load_raw_routes_through_download_and_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UpstreamTwoFile._load_raw must fetch the ROLE's file via _download_and_verify."""
    import safetensors.numpy

    import mlx_taef.kernels._conversion as conv
    from mlx_taef.kernels._conversion import UpstreamTwoFile
    from mlx_taef.kernels._types import WeightSource

    captured: dict[str, object] = {}

    def _fake_dav(source: object, filename: str, *, role: str) -> Path:
        captured["filename"] = filename
        captured["role"] = role
        return Path("/tmp/fake.safetensors")

    monkeypatch.setattr(conv, "_download_and_verify", _fake_dav)
    monkeypatch.setattr(safetensors.numpy, "load_file", lambda p: {"raw": 1})

    src = WeightSource(
        repo="x/y", decoder_filename="dec.safetensors", encoder_filename="enc.safetensors"
    )
    out = UpstreamTwoFile()._load_raw(src, "encoder")
    assert captured["filename"] == "enc.safetensors"  # role-selected file routed through helper
    assert captured["role"] == "encoder"
    assert out == {"raw": 1}


def test_unpinned_source_cache_key_includes_converter_version() -> None:
    from mlx_taef.kernels._types import WeightSource

    src = WeightSource(repo="madebyollin/taef1", filename="diffusion_pytorch_model.safetensors")
    assert src.cache_key(role="decoder") == (
        "madebyollin_taef1__diffusion_pytorch_model.safetensors__decoder__converter-v1"
    )


def test_cache_key_changes_when_revision_changes() -> None:
    from mlx_taef.kernels._types import WeightSource

    a = WeightSource(repo="x/y", filename="w.safetensors", revision="a" * 40)
    b = WeightSource(repo="x/y", filename="w.safetensors", revision="b" * 40)
    assert a.cache_key(role="decoder") != b.cache_key(role="decoder")


def test_cache_key_changes_when_sha_changes() -> None:
    from mlx_taef.kernels._types import WeightSource

    a = WeightSource(repo="x/y", filename="w.safetensors", sha256="a" * 64)
    b = WeightSource(repo="x/y", filename="w.safetensors", sha256="b" * 64)
    assert a.cache_key(role="decoder") != b.cache_key(role="decoder")


def test_cache_key_stays_path_safe_with_pins() -> None:
    """The revision/sha suffix must not reintroduce path-escape characters."""
    from mlx_taef.kernels._types import WeightSource

    src = WeightSource(
        repo="ionden/taew2.1",
        filename="taew2_1.safetensors",
        revision="2ac5ae1c3291a8607a2d6c423b9a0337cef45f2b",
        sha256="04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f",
    )
    key = src.cache_key(role="decoder")
    assert "/" not in key
    assert ".." not in key
    assert "rev-2ac5ae1c3291" in key
    assert "sha-04766eac0221" in key


def test_weightsource_accepts_complete_role_sha256_pair() -> None:
    src = WeightSource(
        repo="x/y",
        decoder_filename="dec.safetensors",
        encoder_filename="enc.safetensors",
        decoder_sha256="a" * 64,
        encoder_sha256="b" * 64,
    )

    assert src.sha256_for("decoder") == "a" * 64
    assert src.sha256_for("encoder") == "b" * 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decoder_sha256": "a" * 64},
        {"encoder_sha256": "b" * 64},
        {"sha256": "c" * 64, "decoder_sha256": "a" * 64, "encoder_sha256": "b" * 64},
    ],
)
def test_weightsource_rejects_partial_or_mixed_role_sha256(kwargs: dict[str, str]) -> None:
    from mlx_taef.kernels._types import WeightSource

    with pytest.raises(ValueError, match="sha256"):
        WeightSource(
            repo="x/y",
            decoder_filename="dec.safetensors",
            encoder_filename="enc.safetensors",
            **kwargs,
        )


def test_single_file_sha256_applies_to_both_roles() -> None:
    src = WeightSource(repo="x/y", filename="both.safetensors", sha256="a" * 64)

    assert src.sha256_for("decoder") == "a" * 64
    assert src.sha256_for("encoder") == "a" * 64


def test_every_registered_source_is_revision_and_digest_pinned() -> None:
    from mlx_taef.kernels import KERNELS

    for name, kernel in KERNELS.items():
        source = kernel.source
        assert source.revision is not None, name
        assert len(source.revision) == 40, name
        assert source.sha256_for("decoder") is not None, name
        assert source.sha256_for("encoder") is not None, name


def test_cache_key_contains_converter_version_and_role_digest() -> None:
    from mlx_taef.kernels._types import CONVERTER_VERSION

    src = WeightSource(
        repo="x/y",
        decoder_filename="dec.safetensors",
        encoder_filename="enc.safetensors",
        decoder_sha256="a" * 64,
        encoder_sha256="b" * 64,
    )

    decoder_key = src.cache_key(role="decoder")
    encoder_key = src.cache_key(role="encoder")
    assert f"converter-v{CONVERTER_VERSION}" in decoder_key
    assert "sha-aaaaaaaaaaaa" in decoder_key
    assert "sha-bbbbbbbbbbbb" in encoder_key


def test_taehv_combined_load_raw_routes_through_download_and_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TaehvCombined._load_raw must fetch via _download_and_verify and select the role."""
    import numpy as np
    import safetensors.numpy

    import mlx_taef.kernels._conversion as conv
    from mlx_taef.kernels._conversion import TaehvCombined
    from mlx_taef.kernels._types import WeightSource

    captured: dict[str, object] = {}

    def _fake_dav(source: object, filename: str, *, role: str) -> Path:
        captured["filename"] = filename
        captured["role"] = role
        return Path("/tmp/fake.safetensors")

    monkeypatch.setattr(conv, "_download_and_verify", _fake_dav)
    monkeypatch.setattr(
        safetensors.numpy,
        "load_file",
        lambda p: {
            "decoder.w": np.ones(2, dtype=np.float16),
            "encoder.w": np.zeros(2, dtype=np.float16),
        },
    )

    src = WeightSource(repo="x/y", filename="combined.safetensors")
    out = TaehvCombined()._load_raw(src, "decoder")
    assert captured["filename"] == "combined.safetensors"
    assert captured["role"] == "decoder"
    assert list(out.keys()) == ["w"]  # role prefix stripped by _select_role
    assert out["w"].dtype == np.float32  # fp16 -> fp32 cast
