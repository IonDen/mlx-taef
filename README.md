# mlx-taef

<p align="center">
  <img src="docs/assets/mlx-taef-logo.png" alt="mlx-taef" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-taef.svg)](https://pypi.org/project/mlx-taef/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-taef.svg)](https://pypi.org/project/mlx-taef/)
[![License: MIT](https://img.shields.io/pypi/l/mlx-taef.svg)](https://github.com/IonDen/mlx-taef/blob/main/LICENSE)

Tiny AutoEncoders for diffusion latents on Apple Silicon, in pure MLX.

`mlx-taef` is the first MLX port of the TAESD family — TAESD (SD1.x), TAESDXL (SDXL), TAEF1 (FLUX.1), TAEF2 (FLUX.2 Klein) — distilled mini-autoencoders that decode diffusion latents to RGB in milliseconds using a few-MB model instead of multi-GB full VAEs.

Use it for:
- **Live previews** during long generations on Mac — TAEF1 decodes a 512×512 preview in ~183 ms and TAEF2 in ~258 ms on M1 Max (vs 2 s for the full VAE). See [COMPARISON.md](COMPARISON.md) for the measured table and reproducer.
- **Low-memory fallbacks** when the full VAE OOMs on 16 GB Macs (TAEF2 peaks at ~0.6 GB decode memory vs ~2.6 GB for the full FLUX.2 VAE on the same latent).
- **Quick latent inspection** in notebooks and ML research.

```python
import mlx.core as mx
from mlx_taef import TAEF2

taef = TAEF2.from_pretrained()              # downloads + converts on first call
img = taef.decode(latents)                  # NHWC float in [0, 1]
img_uint8 = taef.decode_image(latents)      # uint8 NHWC ready for PIL
```

## Which library do I need?

**You want live previews or low-memory FLUX decode?** You're in the right place. `mlx-taef` decodes diffusion latents to RGB in ~260 ms (TAEF2) or ~185 ms (TAEF1) on M1 Max — vs ~2 seconds for the full VAE, with ~4× less peak memory. Drops into mflux via `LivePreviewCallback`.

**You want FLUX generation itself to be faster on Apple Silicon?** You want [`mlx-teacache`](https://github.com/IonDen/mlx-teacache) — it skips redundant denoising steps when the schedule is cacheable (measured 1.46× on FLUX.1-dev at 25 steps).

**You want both: faster generation AND live previews?** Use them together — they compose cleanly. mflux 4-step Klein + TeaCache + TAEF2 previews = 1.30× wall-clock and 26% less peak memory vs vanilla.

## Install

From PyPI:

```bash
pip install mlx-taef
# With the mflux preview callback:
pip install "mlx-taef[mflux]"
```

Or with `uv`:

```bash
uv add mlx-taef
# With mflux:
uv add "mlx-taef[mflux]"
```

Pin an exact version in a project that needs reproducibility:

```bash
pip install "mlx-taef==0.2.0"
```

Verify the install:

```bash
mlx-taef --help
```

Requires Python ≥ 3.11 and Apple Silicon (`mlx` itself is Apple-Silicon-only). Runtime install has **zero PyTorch dependency** — `torch` is dev-only and used solely for fixture generation in the test suite.

## Variants

| Variant | latent_channels | For | HF source |
|---|---|---|---|
| `TAESD` | 4 | Stable Diffusion 1.x | [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) |
| `TAESDXL` | 4 | Stable Diffusion XL | [madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl) |
| `TAEF1` | 16 | FLUX.1 | [madebyollin/taef1](https://huggingface.co/madebyollin/taef1) |
| `TAEF2` | 32 | FLUX.2 Klein | [madebyollin/taef2](https://huggingface.co/madebyollin/taef2) |

All four share one API.

## Benchmarks

Side-by-side images + measured timings: see [COMPARISON.md](COMPARISON.md).

All numbers there come from `scripts/run_showcase.py` (subprocess-per-rep bench harness) and the committed `_artifacts/showcase_report.json`. Per-rep raw arrays are preserved so reviewers can see variance, not just summary stats.

The previous v0.1.x README claim — *"~100 ms decode at 1024×1024, 50–100× faster than the full Flux VAE; ~1 GB peak vs ~9.6 GB"* — was a same-process measurement under v0.1's `tests/test_perf.py`. v0.2.0 re-measures under subprocess-per-rep with per-condition memory caps; see COMPARISON.md for the honest replacement numbers.

## mflux live previews

```python
from mflux.models.flux2 import Flux2Klein
from mlx_taef.integrations.mflux import LivePreviewCallback

model = Flux2Klein.from_pretrained("4bit")
preview = LivePreviewCallback(
    flux=model,            # auto-extracts the Flux2VAE BN stats for exact color
    variant="taef2",
    every=5,
    save_to="preview.png",
    latent_height=32,      # 512 / 16
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

Passing `flux=model` lets the callback auto-extract `model.vae.bn.running_mean` and `running_var` so TAEF2 previews are color-correct out of the box (`callback.resolved_bn == "auto"`). If you have a custom integration where `flux=` isn't convenient, pass `bn_mean=` and `bn_var=` explicitly — those take precedence (`resolved_bn == "explicit"`). Without either path you get identity-BN previews with correct structure but shifted colors (`resolved_bn == "none"`).

See `docs/manual-verification.md` for the full verification recipe.

## Status

- **v0.1.0 — initial public release on PyPI** (2026-05-13). All four variants, encoder + decoder, mflux integration, CI, 99 % honest coverage.
- **v0.2.0 — released on PyPI** (2026-05-27). Auto-bn extraction in `LivePreviewCallback(flux=...)`; per-step gallery mode (`numbered_frames=True`); subprocess-per-rep showcase bench (`scripts/run_showcase.py`); hardware-aware memory caps via `mlx_taef._memory_caps`; [COMPARISON.md](COMPARISON.md) + committed JSON report; [ROADMAP.md](ROADMAP.md).
- **v0.2.3 — released on PyPI** (2026-05-29). Weight loading is now strict: `from_pretrained_local` raises on an incomplete or wrong-shaped weights file instead of loading a silently-wrong model, and the HF→MLX converter checks parameter coverage and shapes at convert time (new `ConversionError`). The end-to-end parity tests now gate on an absolute pixel tolerance rather than cosine similarity. A bare `pytest` skips the network and benchmark tests by default (`--run-network` / `--run-benchmark` to opt in).

Track future releases via the [PyPI history](https://pypi.org/project/mlx-taef/#history) or `gh release list -R IonDen/mlx-taef`.

## License

MIT. Mirrors upstream [madebyollin/taesd](https://github.com/madebyollin/taesd) license. Pretrained weights belong to their respective authors (madebyollin).

## Acknowledgements

- [madebyollin](https://github.com/madebyollin) for the upstream TAESD-family models and weights.
- [Apple ML Explore](https://github.com/ml-explore/mlx) for MLX.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) for the MLX-native FLUX runner this library integrates with.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
