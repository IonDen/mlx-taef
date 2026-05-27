"""Decode a FLUX latent with TAEF2 (no mflux generation needed).

Target search query: "decode FLUX latents MLX", "TAEF2 standalone".
Expected output: writes `out.webp` (512x512 RGB) next to this script.

Run with:
    uv run python examples/decode_flux_latent.py

Loads the committed showcase latent fixture
(tests/fixtures/showcase_latents/flux2_klein_base_4b.safetensors),
unpacks it via the mflux integration helper, and decodes with TAEF2.
Demonstrates the non-mflux entry point — useful for offline latent
inspection or pipeline integration without running the full diffusion
loop.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_taef import TAEF2
from mlx_taef.integrations.mflux import unpack_flux2_latent

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "showcase_latents" / "flux2_klein_base_4b.safetensors"
OUT_PATH = Path(__file__).resolve().parent / "out.webp"


def main() -> None:
    print(f"loading fixture: {FIXTURE}")
    arrays = mx.load(str(FIXTURE))
    latent = arrays["latent"]
    height = int(arrays["height"].item())
    width = int(arrays["width"].item())
    bn_mean = arrays["bn_mean"]
    bn_var = arrays["bn_var"]
    print(f"  latent shape: {latent.shape}, target image: {height}x{width}")

    print("unpacking + decoding with TAEF2...")
    taef = TAEF2.from_pretrained(include_encoder=False)
    nhwc = unpack_flux2_latent(
        latent,
        latent_height=height // 16,
        latent_width=width // 16,
        bn_mean=bn_mean,
        bn_var=bn_var,
    )
    img_uint8 = taef.decode_image(nhwc)
    mx.eval(img_uint8)

    Image.fromarray(np.array(img_uint8[0])).save(OUT_PATH, "WEBP", quality=92)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
