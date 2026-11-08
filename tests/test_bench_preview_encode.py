"""Tests for scripts/bench_preview_encode.py: the pure parts, with a fake clock, no real timing."""

import hashlib
import io
import itertools

import numpy as np
import pytest
from PIL import Image
from scripts import bench_preview_encode as bench


def test_defaults_are_nine_reps_and_no_output_file() -> None:
    """Catches: a default rep count other than the 9 the usage line documents."""
    args = bench.parse_args([])
    assert args.reps == 9
    assert args.out is None


@pytest.mark.parametrize("reps", ["6", "0", "-1"])
def test_fewer_than_seven_reps_are_rejected(reps: str) -> None:
    """Catches: the reps floor checked with `<` against 6, or not at all."""
    with pytest.raises(SystemExit):
        bench.parse_args(["--reps", reps])


def test_seven_reps_and_an_output_path_are_accepted() -> None:
    args = bench.parse_args(["--reps", "7", "--out", "r.json"])
    assert args.reps == 7
    assert str(args.out) == "r.json"


def test_summary_reports_median_and_min_in_milliseconds() -> None:
    """Catches: seconds reported as milliseconds, or the mean reported as the median."""
    summary = bench.summarize_ms([0.030, 0.010, 0.011])
    assert summary == {"median_ms": 11.0, "min_ms": 10.0, "max_ms": 30.0, "reps": 3}


def test_summary_median_of_an_even_count_is_the_midpoint() -> None:
    assert bench.summarize_ms([0.004, 0.001, 0.002, 0.003])["median_ms"] == 2.5


def test_a_frame_whose_hash_differs_from_the_pin_is_refused() -> None:
    """Catches: the hash check skipped or comparing the wrong digest, so a changed frame would
    silently produce numbers for a different image."""
    data = b"some frame bytes"
    bench.verify_sha256(data, hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="sha256"):
        bench.verify_sha256(data + b"!", hashlib.sha256(data).hexdigest())


def test_the_pinned_frame_loads_as_rgb_uint8_and_matches_its_pin() -> None:
    """Catches: the pinned path or hash going stale, or a frame left in RGBA/YUV form."""
    frame = bench.load_frame()
    assert frame.dtype == np.uint8
    assert frame.shape == (512, 512, 3)


def test_png_encode_is_lossless_png_at_each_level() -> None:
    """Catches: the encode writing another format, or a level that is not applied."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    for level in (6, 1):
        data = bench.encode_png(arr, level)
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert np.array_equal(np.array(Image.open(io.BytesIO(data))), arr)
    smooth = np.zeros((64, 64, 3), dtype=np.uint8)
    assert len(bench.encode_png(smooth, 0)) > len(bench.encode_png(smooth, 9))


def test_timing_excludes_warmup_and_interleaves_the_levels() -> None:
    """Catches: warmup reps counted in the stats, or all reps of one level run back to back
    (which lets thermal drift favour one level)."""
    ticks = itertools.count()
    order: list[int] = []

    def _encode(arr: np.ndarray, level: int) -> bytes:
        order.append(level)
        return b"x" * level

    samples, sizes = bench.time_encodes(
        np.zeros((2, 2, 3), dtype=np.uint8),
        levels=(6, 1),
        warmup=2,
        reps=3,
        encode=_encode,
        clock=lambda: float(next(ticks)),
    )

    assert order[:4] == [6, 1, 6, 1]  # warmup
    assert order[4:] == [6, 1, 1, 6, 6, 1]  # order alternates per rep
    assert {level: len(s) for level, s in samples.items()} == {6: 3, 1: 3}
    assert all(s == 1.0 for level_samples in samples.values() for s in level_samples)
    assert sizes == {6: 6, 1: 1}


@pytest.mark.parametrize(
    ("median_l6", "median_l1", "expected"),
    [
        (40.0, 20.0, True),  # exactly half, saving exactly 20 ms
        (40.0, 20.1, False),  # just over half
        (20.0, 10.0, True),  # half, saving exactly 10 ms
        (19.8, 9.9, False),  # half, saving 9.9 ms
        (20.2, 10.1, True),  # half, saving 10.1 ms
        (100.0, 10.0, True),
        (8.0, 1.0, False),  # big ratio, tiny absolute saving
    ],
)
def test_level_1_passes_at_half_the_time_and_ten_ms_saved(
    median_l6: float, median_l1: float, expected: bool
) -> None:
    """Catches: `and` swapped for `or`, `<=` for `<`, or the 10 ms floor shifted."""
    assert bench.level_1_passes(median_l6, median_l1) is expected


_UPSCALED = "lanczos_upscaled_1024x1024"
_NOISY = "lanczos_upscaled_1024x1024_plus_noise_sigma4"


def test_the_level_1_check_uses_the_upscaled_frame_and_reports_every_condition() -> None:
    """Catches: the check read off the native 512x512 frame or the noisy one instead of the
    upscaled 1024x1024 frame, or a per-condition result that does not match its medians."""
    results = {
        "native_512x512": {"6": {"median_ms": 15.6}, "1": {"median_ms": 6.3}},
        _UPSCALED: {"6": {"median_ms": 62.6}, "1": {"median_ms": 21.4}},
        _NOISY: {"6": {"median_ms": 58.5}, "1": {"median_ms": 38.7}},
    }

    check = bench.level_1_check(results)

    assert check["condition"] == _UPSCALED
    assert check["median_l6_ms"] == 62.6
    assert check["median_l1_ms"] == 21.4
    assert check["passes"] is True
    assert check["by_condition"] == {"native_512x512": False, _UPSCALED: True, _NOISY: False}


def test_noise_is_seeded_gaussian_sigma_4_and_clipped_to_uint8() -> None:
    """Catches: unseeded noise (a different frame every run), a sigma other than 4 (a variance
    of 4 gives a spread of 2), or values wrapped around instead of clipped at 0 and 255."""
    flat = np.full((512, 512, 3), 128, dtype=np.uint8)

    noisy = bench.add_noise(flat, seed=0)

    assert noisy.dtype == np.uint8
    assert noisy.shape == flat.shape
    assert np.array_equal(noisy, bench.add_noise(flat, seed=0))
    assert not np.array_equal(noisy, bench.add_noise(flat, seed=1))
    spread = float(np.std(noisy.astype(np.float64) - 128.0))
    assert abs(spread - 4.0) < 0.05
    dark = bench.add_noise(np.zeros((64, 64, 3), dtype=np.uint8), seed=0)
    bright = bench.add_noise(np.full((64, 64, 3), 255, dtype=np.uint8), seed=0)
    assert dark.min() == 0
    assert dark.max() < 32  # wrapped negatives would land near 255
    assert bright.max() == 255
    assert bright.min() > 223  # wrapped overflow would land near 0


def test_run_measures_native_upscaled_and_noisy_frames_and_records_the_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: the noisy condition missing, built from the wrong frame or seed, or measured on
    the plain upscaled frame; the chip or memory missing from the recorded environment."""
    measured: dict[str, np.ndarray] = {}
    timings = {
        (512, 512): {"6": {"median_ms": 15.6}, "1": {"median_ms": 6.3}},
        (1024, 1024): {"6": {"median_ms": 62.6}, "1": {"median_ms": 21.4}},
    }

    def _measure(arr: np.ndarray, reps: int) -> dict[str, dict[str, float | int]]:
        key = f"call{len(measured)}"
        measured[key] = arr
        return timings[arr.shape[:2]]  # type: ignore[index,return-value]

    monkeypatch.setattr(bench, "_measure", _measure)
    monkeypatch.setattr(
        bench,
        "read_sysctl",
        {"machdep.cpu.brand_string": "Apple M9", "hw.memsize": "34359738368"}.get,
    )

    result = bench.run(7)

    assert list(result["results"]) == ["native_512x512", _UPSCALED, _NOISY]
    native, upscaled, noisy = measured.values()
    assert native.shape == (512, 512, 3)
    assert upscaled.shape == (1024, 1024, 3)
    assert np.array_equal(noisy, bench.add_noise(upscaled, seed=0))
    assert result["noise"] == {"seed": 0, "sigma": 4.0, "applied_to": _UPSCALED}
    assert result["level_1_check"]["condition"] == _UPSCALED
    assert result["env"]["chip"] == "Apple M9"
    assert result["env"]["memory_gib"] == 32.0


class _Completed:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_Completed(0, "Apple M1 Max\n"), "Apple M1 Max"),
        (_Completed(1, "sysctl: unknown oid\n"), ""),
        (FileNotFoundError("sysctl"), ""),
    ],
    ids=["ok", "nonzero-exit", "no-sysctl"],
)
def test_read_sysctl_returns_the_value_or_an_empty_string(outcome: object, expected: str) -> None:
    """Catches: a failed or missing `sysctl` crashing the bench, or error text recorded as the
    chip name."""
    calls: list[list[str]] = []

    def _runner(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    assert bench.read_sysctl("machdep.cpu.brand_string", runner=_runner) == expected
    assert calls == [["sysctl", "-n", "machdep.cpu.brand_string"]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("34359738368", 32.0),
        ("17179869184\n", 16.0),
        ("25769803776", 24.0),
        ("", None),
        ("x", None),
    ],
)
def test_memory_is_reported_in_gib(text: str, expected: float | None) -> None:
    """Catches: bytes reported as GB (10**9, giving 34.4 for 32 GiB) or an unreadable value
    crashing the bench."""
    assert bench.memory_gib(text) == expected
