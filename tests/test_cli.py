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


def test_cli_info_reports_missing_file_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nonexistent = tmp_path / "does_not_exist.safetensors"
    assert main(["info", str(nonexistent)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "does_not_exist.safetensors" in captured.err


def test_cli_info_reports_invalid_safetensors_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = tmp_path / "invalid.safetensors"
    invalid.write_bytes(b"not safetensors")

    assert main(["info", str(invalid)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "invalid.safetensors" in captured.err


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


def test_cli_convert_excludes_zimage_choice(capsys: pytest.CaptureFixture[str]) -> None:
    """convert routes through the legacy shim; zimage (no distinct weights) must not be offered.

    Use an invalid value that ISN'T 'zimage' so argparse doesn't echo 'zimage' as the rejected
    value — then the only way 'zimage' could appear is in the listed valid choices, which it must not.
    """
    with pytest.raises(SystemExit):
        main(["convert", "--variant", "not-a-variant", "--dst", "/tmp/x.safetensors"])
    err = capsys.readouterr().err
    assert "zimage" not in err  # zimage is not among convert's valid choices
    for name in ("taesd", "taesdxl", "taef1", "taef2"):
        assert name in err


def test_cli_bench_includes_zimage_choice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["bench", "--variant", "not-a-variant"])
    assert "zimage" in capsys.readouterr().err


def test_bench_cls_by_name_covers_every_kernel() -> None:
    """Behavioral guard: the bench class map must cover all KERNELS (no KeyError, no Phase-3 drop)."""
    from mlx_taef.cli import _BENCH_CLS_BY_NAME
    from mlx_taef.kernels import KERNELS

    assert set(_BENCH_CLS_BY_NAME) == set(KERNELS)


def test_cli_convert_routes_role_to_correct_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
