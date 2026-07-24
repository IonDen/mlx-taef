# Examples

Worked examples of mlx-taef in use. Most include real captured frames and the measured cost of
each decode; the Qwen-Image example is decode-verified against committed parity fixtures, with its
frames and live-preview timing still pending (that section explains why). The point of a tiny
autoencoder is to watch a diffusion run progress without paying for the full VAE on every step, so
most of these are live-preview walkthroughs.

Every measured number below was produced by the committed bench harness and is reproducible with
the command shown in its section, except where a section marks its figures as pending or
community-measured. Captures and timings: Apple M1 Max, 32 GB unified memory, macOS;
mflux 0.18.0, MLX 0.31.2; weights quantized to int4 (`quantize=4`), bf16 generation and fp32 decode. Decode
times measure the decode step in isolation, outside of model construction and after one untimed
warmup call, as the median over several timed reps — the steady-state per-step cost a live preview
pays after its first step. SSIM compares the tiny-decoder image against the full VAE on the same
latent. Captured for mlx-taef v0.7.0 at commit `1e79c29` on 2026-07-24.

Runnable scripts live in [`examples/`](examples/).

## Z-Image-Turbo live preview

Z-Image-Turbo's VAE shares FLUX.1's 16-channel latent contract, so the existing TAEF1 decoder
previews it with no new weights to download. This is the fastest preview in the set.

Recipe: Z-Image-Turbo, prompt `"a red apple on a wooden table"`, seed 42, 512×512, 4 steps,
guidance 0.0 (Turbo is distilled and ignores guidance), int4. Script:
[`examples/zimage_live_preview.py`](examples/zimage_live_preview.py).

Mid-denoise previews (steps 1, 3, final):

| step 1 | step 3 | final |
|---|---|---|
| ![z1](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step01.webp) | ![z3](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step03.webp) | ![zf](_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp) |

What happened: the callback decodes the in-flight latent every step with TAEF1 and writes a
preview frame, while mflux finishes the run and decodes the final image with the full Z-Image
VAE. By step 1 the composition is already readable.

The gain: decoding the final Z-Image latent takes **30 ms** with TAEF1 versus **0.24 s** with the
full Z-Image VAE, about **8.0× faster**, and it peaks at **0.55 GB** instead of 2.61 GB. Structural
similarity between the two is **SSIM 0.94**, so the preview tracks the real image closely. The
TAEF1 decoder is a few MB; the full VAE is hundreds. Reproduce:

```
uv run python scripts/run_showcase.py --scenario zimage_vs_vae
```

A note on scope: Z-Image support is validated for decode and live preview. The showcase command
above measures the SSIM number, and an opt-in network test (`pytest --run-network`) guards the
≥ 0.75 contract against the full Z-Image VAE; neither runs in default CI. `ZImage` inherits the
full API including `encode()`, which reuses the TAEF1 encoder on the shared latent contract. It's fine for round-tripping but not separately validated
against Z-Image's own VAE encoder, so treat encode/img2img as best-effort.

## Qwen-Image live preview

Qwen-Image and Qwen-Image-Edit generate in the Wan 2.1 VAE's 16-channel latent space, which the
rest of the TAESD family doesn't cover. `QwenImage` ports madebyollin's taew2.1 tiny autoencoder
for it, so `LivePreviewCallback(variant="qwen-image")` previews an in-flight Qwen generation the
same way the other variants do:

```python
from mlx_taef.integrations.mflux import LivePreviewCallback

callback = LivePreviewCallback(variant="qwen-image", save_to="preview.png", every=5)
# pass `callback` to your mflux Qwen-Image generation
```

Correctness is gated by committed parity fixtures: the decode and encode paths match the upstream
taew2.1 reference to within ~3e-6 (fp32).

Frames and decode timing here are pending. Qwen-Image is a ~20B model that doesn't fit a usable
resolution on 32 GB, so the in-context live preview and the `mlx-taef bench --variant qwen-image`
number are community-measured rather than captured on this reference machine.

## FLUX.2 Klein live preview

A live preview of FLUX.2 Klein with `auto_bn` color correction. Pass `flux=model` and the
callback reads the VAE's batch-norm stats so the previews are color-correct from the first step.

Recipe: FLUX.2 Klein base 4B, same prompt/seed, 512×512, 4 steps, guidance 1.0, int4. Script:
[`examples/mflux_live_preview.py`](examples/mflux_live_preview.py).

| step 1 | step 3 | final |
|---|---|---|
| ![f21](_artifacts/showcase/live_preview/live_preview_step01.webp) | ![f23](_artifacts/showcase/live_preview/live_preview_step03.webp) | ![f2f](_artifacts/showcase/live_preview/live_preview_final.webp) |

The gain: TAEF2 decodes a Klein latent in **31 ms** versus **0.28 s** for the full FLUX.2 VAE
(~9.3× faster), at **0.59 GB** versus 2.80 GB peak. SSIM here is **0.616**, lower than the FLUX.1
family because TAEF2 is a 4 MB preview decoder standing in for a ~340 MB VAE: it keeps the
structure and color and fudges fine detail. That is the deliberate trade for a real-time preview;
reach for the full VAE when you need final-quality fidelity. Reproduce:

```
uv run python scripts/run_showcase.py --scenario taef2_vs_vae
```

## FLUX.1 fast decode

TAEF1's architecture is closer to the FLUX.1 VAE it shadows, so its previews are higher fidelity
than the FLUX.2 pair. Same red-apple latent, two decoders:

| Full FLUX.1 VAE | TAEF1 |
|---|---|
| ![f1v](_artifacts/showcase/taef1/vae/vanilla_vae_rep0.webp) | ![f1t](_artifacts/showcase/taef1/taef/taef1_rep0.webp) |

TAEF1 decodes the same latent in **30 ms** versus **0.30 s** for the full FLUX.1 VAE (~10.0× faster),
at **0.55 GB** versus 3.70 GB peak and **SSIM 0.94**. Reproduce:

```
uv run python scripts/run_showcase.py --scenario taef1_vs_vae
```

## Low-memory decode

The memory story is the other half of the point. Across all three models the tiny decoder peaks
at roughly half a gigabyte (0.55–0.59 GB), while the full VAEs run 2.6–3.7 GB. On a 32 GB Mac shared
between the OS, the diffusion model, and the decoder, that headroom is what lets a preview run
alongside generation without tipping into swap. If you only need to inspect a latent rather than
ship a final image, decoding it with the matching TAEF variant avoids loading the multi-GB VAE.

## Combined with TeaCache

The preview callback and [mlx-teacache](https://github.com/IonDen/mlx-teacache) coexist on the
same mflux callback registry, so you can cache transformer steps and watch the preview at once.

| step 1 | step 3 | final |
|---|---|---|
| ![c1](_artifacts/showcase/combined/combined_step01.webp) | ![c3](_artifacts/showcase/combined/combined_step03.webp) | ![cf](_artifacts/showcase/combined/combined_final.webp) |

Recipe: FLUX.2 Klein base 4B with TeaCache applied, TAEF2 live preview, same prompt/seed/steps.
Reproduce:

```
uv run python scripts/run_showcase.py --scenario combined
```
