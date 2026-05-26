"""Plumbing tests for scripts/bench_decode.py.

MLX-heavy paths mocked at the network/model-load boundary. Sentinel
parsing, JSON schema, dispatch table run for real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_argparse_orchestrator_mode() -> None:
    from scripts.bench_decode import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args([
        "--latent", "/tmp/x.safetensors",
        "--condition", "taef2",
        "--reps", "5",
    ])
    assert args.latent == Path("/tmp/x.safetensors")
    assert args.condition == "taef2"
    assert args.reps == 5
    assert not args.worker_mode


def test_argparse_worker_mode() -> None:
    from scripts.bench_decode import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args([
        "--worker-mode",
        "--latent", "/tmp/x.safetensors",
        "--condition", "taef2",
        "--rep", "0",
        "--save-to", "/tmp/out.webp",
        "--applied-cap-gb", "2",
    ])
    assert args.worker_mode
    assert args.rep == 0
    assert args.save_to == Path("/tmp/out.webp")
    assert args.applied_cap_gb == 2


def test_sentinel_emission_is_first_of_line() -> None:
    from scripts.bench_decode import _emit_sentinel

    result = {
        "condition": "taef2",
        "rep": 0,
        "elapsed_s": 0.1,
        "peak_memory_gb": 1.5,
        "applied_cap_gb": 2,
        "image_path": "/tmp/out.webp",
    }
    line = _emit_sentinel(result)
    assert line.startswith("::BENCH_RESULT::")
    assert "\n" not in line  # one-liner
    payload = json.loads(line[len("::BENCH_RESULT::"):])
    assert payload == result


def test_parse_sentinel_extracts_first_of_line(tmp_path: Path) -> None:
    from scripts.bench_decode import _parse_worker_stdout

    stdout = "\n".join([
        "Loading model...",
        "Decode: 0.094s",
        "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0, "elapsed_s": 0.094}),
        "Done.",
    ])
    parsed = _parse_worker_stdout(stdout)
    assert parsed["condition"] == "taef2"
    assert parsed["rep"] == 0


def test_parse_sentinel_ignores_mid_line_occurrences() -> None:
    """The sentinel string can appear inside a debug print without being
    a sentinel — parser must require line-start."""
    from scripts.bench_decode import _parse_worker_stdout

    stdout = "\n".join([
        "Debug log mentioning ::BENCH_RESULT:: in passing",
        "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0, "elapsed_s": 0.094}),
    ])
    parsed = _parse_worker_stdout(stdout)
    assert parsed["rep"] == 0  # picked the line-start one


def test_parse_sentinel_raises_on_multiple_sentinels() -> None:
    """Deliberate divergence from mlx-teacache: exactly one sentinel per
    worker. Multiple is a contract violation."""
    from scripts.bench_decode import _parse_worker_stdout

    from mlx_taef.errors import TaefError

    stdout = "\n".join([
        "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0}),
        "::BENCH_RESULT::" + json.dumps({"condition": "taef2", "rep": 0}),
    ])
    with pytest.raises(TaefError, match="multiple sentinels"):
        _parse_worker_stdout(stdout)


def test_parse_sentinel_raises_on_missing_sentinel() -> None:
    from scripts.bench_decode import _parse_worker_stdout

    from mlx_taef.errors import TaefError

    stdout = "Just some output, no sentinel."
    with pytest.raises(TaefError, match="no sentinel"):
        _parse_worker_stdout(stdout)


def test_orchestrator_dispatch_per_condition_cap_split() -> None:
    """TAEF worker gets variant memory cap; full-VAE worker gets FULL_VAE_CAP_GB."""
    from scripts.bench_decode import _resolve_cap_gb

    assert _resolve_cap_gb(condition="taef2") == 2
    assert _resolve_cap_gb(condition="taef1") == 1
    assert _resolve_cap_gb(condition="vanilla_vae", flux_variant="flux1-dev") == 6
    assert _resolve_cap_gb(condition="vanilla_vae", flux_variant="flux2-klein-base-4b") == 12


def test_orchestrator_records_failed_rep_and_continues() -> None:
    """When a worker subprocess fails, the rep is recorded with the
    error and the orchestrator moves on."""
    from scripts.bench_decode import _run_orchestrator

    with patch("scripts.bench_decode._run_one_rep") as mock_rep:
        # Rep 0 fails, rep 1 succeeds, rep 2 succeeds.
        mock_rep.side_effect = [
            {"condition": "taef2", "rep": 0, "status": "failed", "error": "OOM"},
            {"condition": "taef2", "rep": 1, "elapsed_s": 0.1, "peak_memory_gb": 1.5},
            {"condition": "taef2", "rep": 2, "elapsed_s": 0.1, "peak_memory_gb": 1.5},
        ]
        result = _run_orchestrator(
            latent_path=Path("/tmp/x.safetensors"),
            condition="taef2",
            reps=3,
            save_dir=Path("/tmp"),
        )

    assert len(result["per_rep_seconds"]) == 2  # only successful reps
    assert len(result["per_rep_failures"]) == 1
    assert result["per_rep_failures"][0]["error"] == "OOM"


def test_orchestrator_exits_nonzero_if_all_reps_fail() -> None:
    """All reps failed → orchestrator must surface the failure."""
    from scripts.bench_decode import _run_orchestrator

    from mlx_taef.errors import TaefError

    with patch("scripts.bench_decode._run_one_rep") as mock_rep:
        mock_rep.side_effect = [
            {"condition": "taef2", "rep": i, "status": "failed", "error": "boom"}
            for i in range(3)
        ]
        with pytest.raises(TaefError, match="all reps failed"):
            _run_orchestrator(
                latent_path=Path("/tmp/x.safetensors"),
                condition="taef2",
                reps=3,
                save_dir=Path("/tmp"),
            )
