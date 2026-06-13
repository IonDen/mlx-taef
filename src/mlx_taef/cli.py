"""mlx-taef command-line interface: convert / info / bench."""

import argparse
import logging
import time
from pathlib import Path

import mlx.core as mx

from mlx_taef.api import TAEF1, TAEF2, TAESD, TAESDXL, ZImage
from mlx_taef.kernels import KERNELS
from mlx_taef.variants import VARIANTS

logger = logging.getLogger(__name__)

_BENCH_CLS_BY_NAME = {
    "taesd": TAESD,
    "taesdxl": TAESDXL,
    "taef1": TAEF1,
    "taef2": TAEF2,
    "zimage": ZImage,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="mlx-taef",
        description="Tiny AutoEncoder family for diffusion on Apple MLX.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert_names = sorted(VARIANTS)  # legacy shim (kernels with a distinct convert path)
    bench_names = sorted(KERNELS)  # all kernels, incl. zimage

    p_convert = sub.add_parser("convert", help="Download upstream weights and convert to MLX")
    p_convert.add_argument("--variant", required=True, choices=convert_names)
    p_convert.add_argument("--role", default="decoder", choices=["decoder", "encoder"])
    p_convert.add_argument("--dst", required=True, type=Path, help="Output .safetensors path")

    p_info = sub.add_parser("info", help="Print info about a converted MLX safetensors file")
    p_info.add_argument("path", type=Path)

    p_bench = sub.add_parser("bench", help="Decode benchmark on the current Mac")
    p_bench.add_argument("--variant", default="taef2", choices=bench_names)

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        return _cmd_convert(args)
    if args.cmd == "info":
        return _cmd_info(args)
    if args.cmd == "bench":  # pragma: no cover
        return _cmd_bench(args)
    return 1  # pragma: no cover


def _cmd_convert(args: argparse.Namespace) -> int:
    from mlx_taef.convert import convert_hf_decoder_to_mlx, convert_hf_encoder_to_mlx
    from mlx_taef.variants import VARIANTS

    config = VARIANTS[args.variant]
    if args.role == "encoder":
        convert_hf_encoder_to_mlx(out_path=args.dst, config=config)
    else:
        convert_hf_decoder_to_mlx(out_path=args.dst, config=config)
    print(f"Wrote {args.dst}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    weights = mx.load(str(args.path))
    print(f"File: {args.path}")
    print(f"Total tensors: {len(weights)}")
    total_params = sum(int(w.size) for w in weights.values())  # type: ignore[misc,union-attr]
    print(f"Total params: {total_params:,}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:  # pragma: no cover
    cls = _BENCH_CLS_BY_NAME[args.variant]

    model = cls.from_pretrained(include_encoder=False)
    # 1024x1024 image with 8x downsample = 128x128 latent
    latent = mx.random.normal((1, 128, 128, cls._kernel.latent.channels)).astype(mx.float16)
    mx.eval(latent)

    # Warm-up
    mx.eval(model.decode(latent))

    times: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        mx.eval(model.decode(latent))
        times.append(time.perf_counter() - start)
    median_ms = sorted(times)[len(times) // 2] * 1000
    print(f"{args.variant} decode median: {median_ms:.1f} ms over {len(times)} runs")
    return 0


__all__ = ["main"]
