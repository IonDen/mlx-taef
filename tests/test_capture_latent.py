"""Plumbing tests for scripts/_capture_latent.py (mflux.generate_image mocked).

Heavy MLX paths are mocked at the network boundary. Output-path logic and
sha256-sidecar generation run for real against tmp_path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest


def test_argparse_accepts_known_variants() -> None:
    from scripts._capture_latent import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "flux1-dev", "--out-dir", "/tmp"])
    assert args.variant == "flux1-dev"
    assert args.out_dir == Path("/tmp")


def test_argparse_rejects_unknown_variant() -> None:
    from scripts._capture_latent import _build_argparser

    parser = _build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "nonexistent", "--out-dir", "/tmp"])


def test_sha256_sidecar_is_correct(tmp_path: Path) -> None:
    from scripts._capture_latent import _write_sha256_sidecar

    target = tmp_path / "fake_latent.safetensors"
    target.write_bytes(b"hello world")
    sidecar = _write_sha256_sidecar(target)

    assert sidecar == target.with_suffix(target.suffix + ".sha256")
    content = sidecar.read_text().strip()
    expected_hash = hashlib.sha256(b"hello world").hexdigest()
    assert content.startswith(expected_hash)


def test_main_writes_safetensors_and_sidecar_flux1(tmp_path: Path) -> None:
    """Heavy mflux path mocked; verify the orchestrator writes both files."""
    import mlx.core as mx
    from scripts import _capture_latent

    fake_latent = mx.zeros((1, 16, 32, 32))

    def _fake_capture(**kwargs: object) -> dict[str, mx.array]:
        return {
            "latent": fake_latent,
            "height": mx.array([kwargs["height"]], dtype=mx.int32),  # type: ignore[arg-type]
            "width": mx.array([kwargs["width"]], dtype=mx.int32),  # type: ignore[arg-type]
        }

    with (
        patch.object(_capture_latent, "_capture", side_effect=_fake_capture),
        patch.object(_capture_latent, "_install_memory_caps"),
    ):
        exit_code = _capture_latent.main(
            [
                "--variant",
                "flux1-dev",
                "--out-dir",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    latent_path = tmp_path / "flux1_dev.safetensors"
    sha_path = tmp_path / "flux1_dev.safetensors.sha256"
    assert latent_path.exists()
    assert sha_path.exists()
    # flux1-dev path stores latent + height + width but NOT bn_mean/bn_var.
    saved = mx.load(str(latent_path))
    assert set(saved.keys()) == {"latent", "height", "width"}


def test_main_persists_bn_stats_for_flux2(tmp_path: Path) -> None:
    """flux2-klein-base-4b path also writes bn_mean + bn_var so downstream
    TAEF2 decoders can reproduce the color-correct output without
    re-loading Flux2Klein."""
    import mlx.core as mx
    from scripts import _capture_latent

    fake_latent = mx.zeros((1, 1024, 128))
    fake_bn_mean = mx.zeros((128,))
    fake_bn_var = mx.ones((128,))

    def _fake_capture(**kwargs: object) -> dict[str, mx.array]:
        return {
            "latent": fake_latent,
            "bn_mean": fake_bn_mean,
            "bn_var": fake_bn_var,
            "height": mx.array([kwargs["height"]], dtype=mx.int32),  # type: ignore[arg-type]
            "width": mx.array([kwargs["width"]], dtype=mx.int32),  # type: ignore[arg-type]
        }

    with (
        patch.object(_capture_latent, "_capture", side_effect=_fake_capture),
        patch.object(_capture_latent, "_install_memory_caps"),
    ):
        exit_code = _capture_latent.main(
            [
                "--variant",
                "flux2-klein-base-4b",
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert exit_code == 0
    latent_path = tmp_path / "flux2_klein_base_4b.safetensors"
    assert latent_path.exists()
    saved = mx.load(str(latent_path))
    assert {"latent", "bn_mean", "bn_var", "height", "width"} <= set(saved.keys())
    assert saved["bn_mean"].shape == (128,)
    assert saved["bn_var"].shape == (128,)
