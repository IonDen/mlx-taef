"""Measure how far a bfloat16 live preview drifts from the float32 one, on a committed latent.

mflux hands the live-preview callback a bfloat16 in-loop latent for Z-Image and FLUX.2 Klein,
and the preview decoders hold float32 weights. This script decodes one committed latent two ways:

- `before`: the bfloat16 latent goes through the unpack and into the decoder as it is, with no
  float32 cast anywhere (the Z-Image unpack is reproduced here without its cast);
- `now`: the latent is cast to float32 and unpacked by the kernel binding, as the callback does.

Both decodes become uint8 RGB through a PNG round trip, as the callback saves them, and the script
prints one JSON line: the case, the latent (repo-relative), the largest and the mean absolute
difference in 255ths over every pixel channel, the MLX version, and the pixel count (height x
width) of each preview.

Offline: the committed latents and the committed tiny decoders under `tests/converted/`; no
download, a few seconds, well under 1 GiB.

Usage:
    uv run python scripts/measure_preview_dtype_shift.py --case zimage
    uv run python scripts/measure_preview_dtype_shift.py --case flux2-klein
"""

import argparse
import importlib.metadata
import io
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx  # noqa: E402  (after sys.path tweak)
import numpy as np  # noqa: E402

from mlx_taef import TAEF2, Taef, ZImage  # noqa: E402
from mlx_taef.errors import TaefError  # noqa: E402
from mlx_taef.kernels import UnpackContext, get_kernel  # noqa: E402
from mlx_taef.kernels.flux import unpack_flux2_latent  # noqa: E402

_LATENTS = Path("tests/fixtures/showcase_latents")
_DECODERS = Path("tests/converted")

# (bfloat16 decoder input, float32 decoder input) for one committed latent file's arrays.
_Inputs = Callable[[dict[str, mx.array]], tuple[mx.array, mx.array]]


def _zimage_inputs(arrays: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    """Z-Image: `(16, 1, h, w)` latent to NHWC, without and with the float32 casts."""
    latent = arrays["latent"]
    # `unpack_zimage_latent` with its float32 cast taken out: (1,16,1,h,w) -> (1,16,h,w) -> NHWC.
    before = mx.transpose(mx.squeeze(mx.expand_dims(latent, axis=0), axis=2), (0, 2, 3, 1))
    binding = get_kernel("zimage").integration
    if binding is None:  # pragma: no cover - the zimage kernel always carries its binding
        raise TaefError("the zimage kernel has no mflux binding")
    now = binding.unpack(latent.astype(mx.float32), UnpackContext(latent_height=0, latent_width=0))
    return before, now


def _flux2_klein_inputs(arrays: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    """FLUX.2 Klein: packed `(1, N, 128)` latent to NHWC, without and with the float32 cast."""
    latent = arrays["latent"]
    binding = get_kernel("taef2").integration
    if binding is None or binding.packed_latent_downscale is None:  # pragma: no cover
        raise TaefError("the taef2 kernel has no packed mflux binding")
    downscale = binding.packed_latent_downscale
    ctx = UnpackContext(
        latent_height=int(cast("int", arrays["height"].item())) // downscale,
        latent_width=int(cast("int", arrays["width"].item())) // downscale,
    )
    return unpack_flux2_latent(latent, ctx), binding.unpack(latent.astype(mx.float32), ctx)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Case:
    latent: Path
    decoder: Path
    model: type[Taef]
    inputs: _Inputs


CASES: dict[str, _Case] = {
    "zimage": _Case(
        latent=_LATENTS / "z_image_turbo.safetensors",
        decoder=_DECODERS / "taef1_decoder.safetensors",
        model=ZImage,
        inputs=_zimage_inputs,
    ),
    "flux2-klein": _Case(
        latent=_LATENTS / "flux2_klein_base_4b.safetensors",
        decoder=_DECODERS / "taef2_decoder.safetensors",
        model=TAEF2,
        inputs=_flux2_klein_inputs,
    ),
}


def _saved_rgb(model: Taef, decoder_input: mx.array) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Decode to uint8 RGB and round-trip it through PNG, as the callback saves a preview."""
    from PIL import Image

    image = model.decode_image(decoder_input)
    mx.eval(image)
    buffer = io.BytesIO()
    Image.fromarray(np.array(image[0])).save(buffer, format="PNG")
    buffer.seek(0)
    return np.array(Image.open(buffer))


def measure(case: str) -> dict[str, object]:
    """Decode `case`'s committed latent the bfloat16 way and the float32 way; compare pixels."""
    spec = CASES[case]
    arrays = mx.load(str(_REPO_ROOT / spec.latent))
    if not isinstance(arrays, dict):  # pragma: no cover - safetensors always loads as a dict
        raise TaefError(f"{spec.latent} did not load as named arrays")
    if arrays["latent"].dtype != mx.bfloat16:
        raise TaefError(f"{spec.latent} holds a {arrays['latent'].dtype} latent, not bfloat16")
    before, now = spec.inputs(arrays)
    if before.dtype != mx.bfloat16 or now.dtype != mx.float32:
        raise TaefError(
            f"decoder inputs are {before.dtype} (before) and {now.dtype} (now); "
            "expected bfloat16 and float32"
        )
    model = spec.model.from_pretrained_local(_REPO_ROOT / spec.decoder)
    from_bf16 = _saved_rgb(model, before).astype(np.int16)
    from_fp32 = _saved_rgb(model, now).astype(np.int16)
    diff = np.abs(from_bf16 - from_fp32)
    return {
        "case": case,
        "latent": spec.latent.as_posix(),
        "max_abs_255": int(diff.max()),
        "mean_abs_255": round(float(diff.mean()), 4),
        "mlx": importlib.metadata.version("mlx"),
        "pixels": int(diff.shape[0] * diff.shape[1]),
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pixel shift between a bfloat16 and a float32 live-preview decode."
    )
    parser.add_argument("--case", required=True, choices=list(CASES), help="Committed latent.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the JSON line for `--case`; argparse exits 2 on an unknown case."""
    args = _build_argparser().parse_args(argv)
    print(json.dumps(measure(args.case)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
