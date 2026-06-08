"""Tests for the CLI."""

from pathlib import Path

import pytest

from mlx_taef.cli import main

CONVERTED_DIR = Path(__file__).parent / "converted"


def test_cli_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_info_prints_total_params(capsys: pytest.CaptureFixture[str]) -> None:
    """Use a pre-baked file rather than hitting the network."""
    converted = CONVERTED_DIR / "taef2_decoder.safetensors"
    rc = main(["info", str(converted)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Total params:" in captured.out
    assert "Total tensors:" in captured.out


def test_cli_info_rejects_missing_file(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.safetensors"
    # mx.load raises RuntimeError on a missing file. Match on our own filename
    # (which mlx includes in the message) rather than mlx's prefix wording, so a
    # future mlx error-message reword doesn't false-fail this test; RuntimeError
    # (vs bare Exception) already excludes unrelated SystemExit/ImportError.
    with pytest.raises(RuntimeError, match="does_not_exist"):
        main(["info", str(nonexistent)])


def test_cli_no_subcommand_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse exits with code 2 when required subcommand is missing."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


@pytest.mark.network
def test_cli_convert_writes_safetensors(tmp_path: Path) -> None:
    out = tmp_path / "out.safetensors"
    rc = main(["convert", "--variant", "taef2", "--dst", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 1000


@pytest.mark.network
def test_cli_bench_prints_decode_time(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["bench", "--variant", "taef2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "decode median:" in captured.out


def test_cli_sources_choices_from_registry_not_all_variants() -> None:
    """Migration target: choices come from KERNELS; the legacy ALL_VARIANTS import is gone."""
    import mlx_taef.cli as c

    assert not hasattr(c, "ALL_VARIANTS")
    assert hasattr(c, "KERNELS")


def test_cli_convert_variant_choices_include_all_kernels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["convert", "--variant", "not-a-variant", "--dst", "/tmp/x.safetensors"])
    err = capsys.readouterr().err
    for name in ("taesd", "taesdxl", "taef1", "taef2"):
        assert name in err


def test_cli_convert_routes_role_to_correct_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Offline: fake the converters and verify --role encoder/decoder dispatch is correct."""
    from mlx_taef.variants import VARIANTS

    calls: list[tuple[str, Path, object]] = []

    def fake_enc(*, out_path: Path, config: object) -> None:
        calls.append(("encoder", out_path, config))

    def fake_dec(*, out_path: Path, config: object) -> None:
        calls.append(("decoder", out_path, config))

    monkeypatch.setattr("mlx_taef.convert.convert_hf_encoder_to_mlx", fake_enc)
    monkeypatch.setattr("mlx_taef.convert.convert_hf_decoder_to_mlx", fake_dec)

    out = tmp_path / "out.safetensors"
    assert main(["convert", "--variant", "taef1", "--role", "encoder", "--dst", str(out)]) == 0
    assert calls == [("encoder", out, VARIANTS["taef1"])]

    calls.clear()
    assert main(["convert", "--variant", "taef1", "--role", "decoder", "--dst", str(out)]) == 0
    assert calls == [("decoder", out, VARIANTS["taef1"])]
