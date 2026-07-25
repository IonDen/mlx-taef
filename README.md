# mlx-taef

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-taef/main/docs/assets/mlx-taef-logo.png" alt="mlx-taef" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-taef.svg)](https://pypi.org/project/mlx-taef/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-taef/)
[![License: MIT](https://img.shields.io/pypi/l/mlx-taef.svg)](https://github.com/IonDen/mlx-taef/blob/main/LICENSE)

Tiny AutoEncoders for diffusion latents on Apple Silicon, in pure MLX.

`mlx-taef` is the first MLX port of the TAESD family — TAESD (SD1.x), TAESDXL (SDXL), TAEF1 (FLUX.1), TAEF2 (FLUX.2 Klein), and Z-Image (which reuses the TAEF1 weights for previews) — plus Qwen-Image / Qwen-Image-Edit on a port of the taew2.1 (Wan 2.1 VAE) autoencoder. All are distilled mini-autoencoders that decode diffusion latents to RGB in milliseconds using a few-MB model instead of multi-GB full VAEs.

Use it for:
- **Live previews** during long generations on Mac — the decode step itself takes ~30 ms for both TAEF1 and TAEF2 on M1 Max, versus ~0.3 s for the full VAE decode (~9–10× faster). [Watch a step-by-step decode](https://github.com/IonDen/mlx-taef/blob/main/PREVIEW.md), or see [COMPARISON.md](COMPARISON.md) for the measured table and reproducer.
- **Low-memory fallbacks** when the full VAE OOMs on 16 GB Macs (TAEF2 peaks at ~0.59 GB decode memory vs ~2.8 GB for the full FLUX.2 VAE on the same latent).
- **Quick latent inspection** in notebooks and ML research.

```python
import mlx.core as mx
from mlx_taef import TAEF2

taef = TAEF2.from_pretrained()              # downloads + converts on first call
img = taef.decode(latents)                  # NHWC float in [0, 1]
img_uint8 = taef.decode_image(latents)      # uint8 NHWC ready for PIL
```

## Which library do I need?

**You want live previews or low-memory FLUX decode?** You're in the right place. `mlx-taef`'s decode step takes ~30 ms on M1 Max, for either TAEF2 or TAEF1 — vs ~0.3 s for the full VAE decode, with ~5× less decode memory. Drops into mflux via `LivePreviewCallback`.

**You want FLUX generation itself to be faster on Apple Silicon?** You want [`mlx-teacache`](https://github.com/IonDen/mlx-teacache) — it skips redundant denoising steps when the schedule is cacheable (measured 1.46× on FLUX.1-dev at 25 steps).

**You want both: faster generation AND live previews?** Use them together. mflux 4-step Klein + TeaCache + TAEF2 previews measured 1.31× faster with 43% less peak memory than the same generation without TeaCache.

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
pip install "mlx-taef==0.7.1"
```

Verify the install:

```bash
mlx-taef --help
```

Requires Python ≥ 3.10 and Apple Silicon (`mlx` itself is Apple-Silicon-only). The base `mlx-taef` install pulls in **no PyTorch**; in this repo `torch` is used only to generate test fixtures. The optional `mflux` extra is a separate case: mflux currently depends on PyTorch, so installing `mlx-taef[mflux]` brings it in.

## Variants

| Variant | latent_channels | For | HF source |
|---|---|---|---|
| `TAESD` | 4 | Stable Diffusion 1.x | [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) |
| `TAESDXL` | 4 | Stable Diffusion XL | [madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl) |
| `TAEF1` | 16 | FLUX.1 | [madebyollin/taef1](https://huggingface.co/madebyollin/taef1) |
| `TAEF2` | 32 | FLUX.2 Klein | [madebyollin/taef2](https://huggingface.co/madebyollin/taef2) |
| `ZImage` | 16 | Z-Image / Z-Image-Turbo (shares the FLUX.1 16-ch latent contract) | reuses [madebyollin/taef1](https://huggingface.co/madebyollin/taef1) |
| `QwenImage` | 16 | Qwen-Image / Qwen-Image-Edit (Wan 2.1 VAE 16-ch latent) | [ionden/taew2.1](https://huggingface.co/ionden/taew2.1) |

They all share one API.

## Examples

[EXAMPLES.md](EXAMPLES.md) walks through live-preview and low-memory decode for each model
with real captured frames, final images, and the measured gain. Runnable companions live in
[`examples/`](examples/).

## Benchmarks

Side-by-side images + measured timings: see [COMPARISON.md](COMPARISON.md).

All numbers there come from `scripts/run_showcase.py` (subprocess-per-condition bench harness) and the committed `_artifacts/showcase_report.json`. Per-rep raw arrays are preserved so reviewers can see variance, not just summary stats.

The previous v0.1.x README claim — *"~100 ms decode at 1024×1024, 50–100× faster than the full Flux VAE; ~1 GB peak vs ~9.6 GB"* — was a same-process measurement under v0.1's `tests/test_perf.py`. v0.2.0 re-measures under subprocess-per-rep with per-condition memory caps; see COMPARISON.md for the honest replacement numbers.

## mflux live previews

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2 import Flux2Klein
from mlx_taef.integrations.mflux import LivePreviewCallback

model = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
preview = LivePreviewCallback(
    flux=model,            # auto-extracts the Flux2VAE BN stats for exact color
    variant="taef2",
    every=5,
    save_to="preview.png",
    # latent_height / latent_width are auto-detected from the generation config;
    # pass both only if you want to override.
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

Preview failures default to `on_error="disable"`: the callback logs one warning, stops previewing for the rest of that generation, and lets mflux finish the image. Use `on_error="raise"` when an integration or test needs fail-fast behavior. Calling the same callback in a later generation resets the disabled state.

See `docs/manual-verification.md` for the full verification recipe.

## Status

- **v0.1.0 — initial public release on PyPI** (2026-05-13). All four variants, encoder + decoder, mflux integration, CI, 99 % honest coverage.
- **v0.2.0 — released on PyPI** (2026-05-27). Auto-bn extraction in `LivePreviewCallback(flux=...)`; per-step gallery mode (`numbered_frames=True`); subprocess-per-rep showcase bench (`scripts/run_showcase.py`); hardware-aware memory caps via `mlx_taef._memory_caps`; [COMPARISON.md](COMPARISON.md) + committed JSON report; [ROADMAP.md](ROADMAP.md).
- **v0.2.3 — released on PyPI** (2026-05-29). Weight loading is now strict: `from_pretrained_local` raises on an incomplete or wrong-shaped weights file instead of loading a silently-wrong model, and the HF→MLX converter checks parameter coverage and shapes at convert time (new `ConversionError`). The end-to-end parity tests now gate on an absolute pixel tolerance rather than cosine similarity. A bare `pytest` skips the network and benchmark tests by default (`--run-network` / `--run-benchmark` to opt in).
- **v0.3.0 — released on PyPI** (2026-06-06). Internal kernel refactor: each variant is now a composable `ModelKernel` (`mlx_taef.kernels`), so adding a model is a self-contained entry; `variants.py` stays a back-compat shim. Ships one user-facing fix — the mflux `LivePreviewCallback` FLUX.1 path fed the packed latent straight to the decoder and produced wrong previews; it now unpacks correctly.
- **v0.3.1 — released on PyPI** (2026-06-08). Hardening: `decode()`/`encode()` raise a clear error when weights haven't been loaded yet, or when a latent has the wrong channel count, instead of returning garbage; importing without mflux installed now raises `MfluxNotInstalledError` (a `TaefError` that is also an `ImportError`).
- **v0.4.0 — released on PyPI** (2026-06-13). Z-Image / Z-Image-Turbo live preview: a new `ZImage` model reuses TAEF1's FLUX.1 weights with no new download (Z-Image shares FLUX.1's 16-channel latent contract), validated by an SSIM ≥ 0.75 calibration against mflux's full Z-Image VAE (measured 0.94). Adds `mlx-taef bench --variant zimage` and a new top-level [EXAMPLES.md](EXAMPLES.md) with captured frames and measured decode numbers.
- **v0.4.1 — released on PyPI** (2026-06-14). Documentation and packaging accuracy pass, no code changes: the Z-Image SSIM calibration is described correctly (it runs as an opt-in network test, not in default CI); the "no PyTorch" note is scoped to the base install (the `mflux` extra pulls in PyTorch via mflux); and the source distribution is now an allowlist, so it no longer ships the test suite without its fixtures or sweeps in local tool state.
- **v0.4.2 — released on PyPI** (2026-06-14). Hardening patch: `decode()` and `encode()` now reject non-4-D inputs with a clear `ValueError` naming the expected NHWC rank, instead of failing deep in the conv stack. Also moves the release workflow's `download-artifact` step to a Node 24 release, and the `mflux_live_preview` example now prints per-step decode timing next to the full-VAE decode (thanks @ianscrivener, #20).
- **v0.5.0 — released on PyPI** (2026-06-18). Live-preview ergonomics: `LivePreviewCallback` auto-detects `latent_height` / `latent_width` from the mflux generation config, so you no longer pass them by hand. Passing both still overrides; passing exactly one now raises. The auto-extracted Flux2VAE batch-norm `eps` is forwarded into the TAEF2 unpack so a non-default `eps` previews faithfully, and `auto_bn=True` on a non-TAEF2 variant logs its no-op instead of staying silent. Three showcase-only exceptions (`SchemaVersionError`, `FixtureLatentMissingError`, `MlxTeacacheNotInstalledError`) are no longer exported from the package root; they remain importable from `mlx_taef.errors`.
- **v0.5.1 — released on PyPI** (2026-06-20). The `mflux` extra now installs against mflux 0.18.x as well as 0.17.x; the previous `<0.18` pin excluded 0.18, so users already on it could not install the extra without downgrading. The live-preview integration is verified against mflux 0.18.0 with no API or behavior change.
- **v0.6.0 — released on PyPI** (2026-06-23). Qwen-Image / Qwen-Image-Edit live preview: a new `QwenImage` model ports madebyollin's taew2.1 tiny autoencoder (for the Wan 2.1 VAE's 16-channel latent) to pure MLX. Adds `LivePreviewCallback(variant="qwen-image")` and `mlx-taef bench --variant qwen-image`. Decode and encode match the upstream taew2.1 reference to ~3e-6 (committed parity fixtures). Live-preview quality against the full Wan VAE is community-measured — Qwen-Image is a ~20B model that won't fit a usable resolution on 32 GB.
- **v0.6.1 — released on PyPI** (2026-06-30). Maintenance: converted-weights cache writes are now atomic (temp file + rename), so an interrupted download/convert can no longer leave a truncated file that later runs trust as valid. Adds Python 3.14 support, sharper public-API types (factories return their concrete subclass; the decoder/encoder role and preview variant are typed literals), and a linked live-preview demo page.
- **v0.6.2 — released on PyPI** (2026-07-09). Hardening and accuracy. `LivePreviewCallback` now rejects `every < 1` and a half-set BN pair, and resets its state between generations. The converted-weights cache invalidates when a source's pinned revision or sha256 changes (Qwen-Image re-converts once on upgrade), and every conversion path enforces those pins. The decode benchmark now measures the decode step at steady state, in isolation from one-time model construction: the tiny decoders run ~30 ms per step (earlier releases reported ~180–260 ms because they timed model construction inside the decode window), a ~8–10× speedup over the full VAE decode. COMPARISON / EXAMPLES are re-measured with two Z-Image scenarios added. Also fixes the mflux quickstart earlier in this README.
- **v0.7.0 — released on PyPI** (2026-07-24). `LivePreviewCallback` gains configurable runtime failure handling: it warns once and disables previews for the current generation by default, while `on_error="raise"` preserves strict behavior. All built-in weight sources now have immutable revisions and role-specific sha256 pins, and `mlx-taef convert` uses the same verified source path as runtime loading. Conversion caches include a format version, and CI installs from the lockfile. The six-scenario benchmark requires every repetition and preview frame, exits nonzero after any scenario failure, and records source and installed-package versions separately. The measured report shows no regression against v0.6.2.
- **v0.7.1 — released on PyPI** (2026-07-25). Python 3.10 is now supported: the floor drops from 3.11 to 3.10 and CI runs the full offline suite on 3.10 through 3.14, so no advertised version is untested. `mlx-teacache` (used by the showcase and test extras, and 3.11-only) installs on 3.11+ only; the runtime dependencies all work on 3.10.

Track future releases via the [PyPI history](https://pypi.org/project/mlx-taef/#history) or `gh release list -R IonDen/mlx-taef`.

## License

MIT. Mirrors upstream [madebyollin/taesd](https://github.com/madebyollin/taesd) license. Pretrained weights belong to their respective authors (madebyollin).

## Acknowledgements

- [madebyollin](https://github.com/madebyollin) for the upstream TAESD-family models and weights.
- [Apple ML Explore](https://github.com/ml-explore/mlx) for MLX.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) for the MLX-native FLUX runner this library integrates with.

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache residual caching to skip redundant FLUX denoising steps.
- [mlx-model-doctor](https://github.com/IonDen/mlx-model-doctor) — validate an MLX / Hugging Face model repo before you load it (config, tokenizer, safetensors, memory).
- [mlx-quant-fidelity](https://github.com/IonDen/mlx-quant-fidelity) — measure how much quality a quantization costs (KL divergence, top-token flips, perplexity delta).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
