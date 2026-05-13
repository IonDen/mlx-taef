# mlx-taef

Tiny AutoEncoders for diffusion latents on Apple Silicon, in pure MLX.

`mlx-taef` is the first MLX port of the TAESD family — TAESD (SD1.x), TAESDXL (SDXL), TAEF1 (FLUX.1), TAEF2 (FLUX.2 Klein) — distilled mini-autoencoders that decode diffusion latents to RGB in milliseconds using a few-MB model instead of multi-GB full VAEs.

Use it for:
- **Live previews** during long generations on Mac — see each step refresh in <100 ms instead of waiting 30 s for the full VAE.
- **Low-memory fallbacks** when the full VAE OOMs on 16 GB Macs (TAEF2 peaks at ~1 GB for 1024×1024 vs ~9.6 GB for the full Flux VAE).
- **Quick latent inspection** in notebooks and ML research.

```python
import mlx.core as mx
from mlx_taef import TAEF2

taef = TAEF2.from_pretrained()              # downloads + converts on first call
img = taef.decode(latents)                  # NHWC float in [0, 1]
img_uint8 = taef.decode_image(latents)      # uint8 NHWC ready for PIL
```

## Install

```bash
pip install mlx-taef
# With mflux preview callback:
pip install "mlx-taef[mflux]"
```

Requires Python ≥ 3.11 and Apple Silicon. Runtime install has **zero PyTorch dependency**.

## Variants

| Variant | latent_channels | For | HF source |
|---|---|---|---|
| `TAESD` | 4 | Stable Diffusion 1.x | [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) |
| `TAESDXL` | 4 | Stable Diffusion XL | [madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl) |
| `TAEF1` | 16 | FLUX.1 | [madebyollin/taef1](https://huggingface.co/madebyollin/taef1) |
| `TAEF2` | 32 | FLUX.2 Klein | [madebyollin/taef2](https://huggingface.co/madebyollin/taef2) |

All four share one API.

## Benchmarks (M1 Max, fp16)

| Metric | TAEF2 (this library) | Full Flux VAE (reference) | Win |
|---|---|---|---|
| Decode latency 1024×1024 | **~100 ms** | seconds | 50–100× |
| Peak unified memory 1024×1024 | **~1 GB** | ~9.6 GB | **9.4×** |
| Output cosine sim vs PyTorch reference | > 0.999 | — | (parity verified) |

Numbers from `tests/test_perf.py` on M1 Max 32 GB. See `notes/phase1-benchmarks.md` for details.

## mflux live previews

```python
from mflux.models.flux2 import Flux2Klein
from mlx_taef.integrations.mflux import LivePreviewCallback

model = Flux2Klein.from_pretrained("4bit")
preview = LivePreviewCallback(
    variant="taef2",
    every=5,
    save_to="preview.png",
    latent_height=32,  # 512 / 16
    latent_width=32,
)
model.callbacks.register(preview)
model.generate_image(
    prompt="a red apple on a wooden table",
    num_inference_steps=25,
    width=512,
    height=512,
    seed=42,
)
```

For exact value-space recovery, also pass `bn_mean=flux2_vae.bn.running_mean, bn_var=flux2_vae.bn.running_var` to the callback. Without them, previews show correct structure but colors may shift.

See `docs/manual-verification.md` for the full verification recipe.

## Status

- v0.1.0 — initial public release. All four variants, encoder + decoder, mflux integration, CI, 100% test coverage.

## License

MIT. Mirrors upstream [madebyollin/taesd](https://github.com/madebyollin/taesd) license. Pretrained weights belong to their respective authors (madebyollin).

## Acknowledgements

- [madebyollin](https://github.com/madebyollin) for the upstream TAESD-family models and weights.
- [Apple ML Explore](https://github.com/ml-explore/mlx) for MLX.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) for the MLX-native FLUX runner this library integrates with.
