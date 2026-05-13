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
    # mlx will raise on load — verify it propagates as a non-zero exit (we accept
    # any non-zero or any exception that exits the process).
    with pytest.raises(Exception):  # noqa: PT011,B017
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
