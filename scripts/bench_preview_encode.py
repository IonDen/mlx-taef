"""Time the PNG encode of one live-preview frame at zlib compress levels 6 and 1.

CPU only: Pillow and NumPy, no MLX import and no model. The frame is a committed preview
from a FLUX.2 Klein 4B run, pinned by sha256. It is measured in three conditions: at its own
size (512x512); upscaled to 1024x1024 (LANCZOS), the default FLUX generation size; and the
same upscaled frame plus seeded Gaussian noise (sigma 4 on the 0-255 scale, clipped). The
upscaled frame is smoother than a real render and is the best case for level 1; the noisy
frame is a textured case, where level 1 saves less. The level-1 check is judged on the
upscaled frame and reported for every condition. Each rep encodes once per level,
alternating the order from rep to rep, after warmup encodes that are not counted.

Usage:
    uv run python scripts/bench_preview_encode.py [--reps 9] [--out result.json]

Prints one JSON object on stdout (and writes it to --out when given).
"""

import argparse
import hashlib
import io
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, features

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRAME_REL = "_artifacts/examples/flux2-klein-4b/flux2-klein-4b_step03.webp"
_FRAME_SHA256 = "ae092131abfaf1ea326654164af992554c8f32d995bd02deebb1d1b013df02e9"
_LEVELS = (6, 1)  # Pillow's PNG default, and the fastest zlib level that still compresses
_WARMUP = 2
_MIN_REPS = 7
_DEFAULT_REPS = 9
_UPSCALED_SIDE = 1024
_UPSCALED = f"lanczos_upscaled_{_UPSCALED_SIDE}x{_UPSCALED_SIDE}"
_NOISE_SIGMA = 4.0
_NOISE_SEED = 0
_NOISY = f"{_UPSCALED}_plus_noise_sigma{_NOISE_SIGMA:g}"
_CHECK_MAX_RATIO = 0.5
_CHECK_MIN_SAVING_MS = 10.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; fewer than seven reps is an error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=_DEFAULT_REPS, help="timed reps per level")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    args = parser.parse_args(argv)
    if args.reps < _MIN_REPS:
        parser.error(f"--reps must be at least {_MIN_REPS}, got {args.reps}")
    return args


def verify_sha256(data: bytes, expected: str) -> None:
    """Raise ValueError unless `data` hashes to `expected`."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        msg = f"frame sha256 is {actual}, expected {expected}"
        raise ValueError(msg)


def load_frame() -> np.ndarray:
    """Load the pinned preview frame as an (H, W, 3) uint8 array, after checking its hash."""
    data = (_REPO_ROOT / _FRAME_REL).read_bytes()
    verify_sha256(data, _FRAME_SHA256)
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def add_noise(arr: np.ndarray, *, seed: int, sigma: float = _NOISE_SIGMA) -> np.ndarray:
    """Return `arr` plus seeded Gaussian noise, rounded and clipped to uint8.

    The noise has standard deviation `sigma` on the 0-255 scale and is drawn from
    `numpy.random.default_rng(seed)`.
    """
    rng = np.random.default_rng(seed)
    noisy = arr.astype(np.float64) + rng.normal(0.0, sigma, arr.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def read_sysctl(name: str, *, runner: Callable[..., Any] = subprocess.run) -> str:
    """Return `sysctl -n <name>`, stripped, or an empty string if it cannot be read."""
    try:
        completed = runner(["sysctl", "-n", name], check=False, capture_output=True, text=True)
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return str(completed.stdout).strip()


def memory_gib(memsize: str) -> float | None:
    """Convert `hw.memsize` (bytes, as text) to GiB; None if it is not a number."""
    try:
        return round(int(memsize) / 2**30, 1)
    except ValueError:
        return None


def encode_png(arr: np.ndarray, level: int) -> bytes:
    """Encode `arr` the way the preview save does, as PNG at zlib level `level`."""
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG", compress_level=level)
    return buffer.getvalue()


def time_encodes(
    arr: np.ndarray,
    *,
    levels: Sequence[int],
    warmup: int,
    reps: int,
    encode: Callable[[np.ndarray, int], bytes] = encode_png,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[int, list[float]], dict[int, int]]:
    """Time `encode` per level; return seconds per rep and encoded bytes, by level."""
    for _ in range(warmup):
        for level in levels:
            encode(arr, level)
    samples: dict[int, list[float]] = {level: [] for level in levels}
    sizes: dict[int, int] = {}
    for rep in range(reps):
        order = levels if rep % 2 == 0 else tuple(reversed(levels))
        for level in order:
            start = clock()
            data = encode(arr, level)
            samples[level].append(clock() - start)
            sizes[level] = len(data)
    return samples, sizes


def summarize_ms(samples_s: Sequence[float]) -> dict[str, float | int]:
    """Median, min and max of `samples_s` (seconds), in milliseconds."""
    return {
        "median_ms": round(statistics.median(samples_s) * 1000, 3),
        "min_ms": round(min(samples_s) * 1000, 3),
        "max_ms": round(max(samples_s) * 1000, 3),
        "reps": len(samples_s),
    }


def level_1_passes(median_l6_ms: float, median_l1_ms: float) -> bool:
    """Level 1 takes at most half the time of level 6 and saves at least 10 ms."""
    return (
        median_l1_ms <= _CHECK_MAX_RATIO * median_l6_ms
        and median_l6_ms - median_l1_ms >= _CHECK_MIN_SAVING_MS
    )


def _passes(timings: Mapping[str, Mapping[str, float | int]]) -> bool:
    return level_1_passes(float(timings["6"]["median_ms"]), float(timings["1"]["median_ms"]))


def level_1_check(
    results: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> dict[str, Any]:
    """Judge level 1 on the upscaled 1024x1024 frame, and report the result for every condition."""
    l6 = float(results[_UPSCALED]["6"]["median_ms"])
    l1 = float(results[_UPSCALED]["1"]["median_ms"])
    return {
        "condition": _UPSCALED,
        "check": "median(L1) <= 0.5 * median(L6) and median(L6) - median(L1) >= 10 ms",
        "median_l6_ms": l6,
        "median_l1_ms": l1,
        "passes": level_1_passes(l6, l1),
        "by_condition": {name: _passes(timings) for name, timings in results.items()},
    }


def _measure(arr: np.ndarray, reps: int) -> dict[str, dict[str, float | int]]:
    samples, sizes = time_encodes(arr, levels=_LEVELS, warmup=_WARMUP, reps=reps)
    return {
        str(level): {**summarize_ms(samples[level]), "bytes": sizes[level]} for level in _LEVELS
    }


def run(reps: int) -> dict[str, Any]:
    """Measure the pinned frame native, upscaled, and upscaled with noise; return the result."""
    frame = load_frame()
    height, width = frame.shape[:2]
    upscaled = np.array(
        Image.fromarray(frame).resize((_UPSCALED_SIDE, _UPSCALED_SIDE), Image.Resampling.LANCZOS)
    )
    noisy = add_noise(upscaled, seed=_NOISE_SEED)
    results = {
        f"native_{width}x{height}": _measure(frame, reps),
        _UPSCALED: _measure(upscaled, reps),
        _NOISY: _measure(noisy, reps),
    }
    return {
        "frame": {"path": _FRAME_REL, "sha256": _FRAME_SHA256, "width": width, "height": height},
        "noise": {"seed": _NOISE_SEED, "sigma": _NOISE_SIGMA, "applied_to": _UPSCALED},
        "levels": list(_LEVELS),
        "warmup": _WARMUP,
        "reps": reps,
        "results": results,
        "level_1_check": level_1_check(results),
        "env": {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "zlib": features.version("zlib"),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "chip": read_sysctl("machdep.cpu.brand_string"),
            "memory_gib": memory_gib(read_sysctl("hw.memsize")),
            "macos": platform.mac_ver()[0],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench, print the JSON, and write it to --out when given."""
    args = parse_args(argv)
    text = json.dumps(run(args.reps), indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
