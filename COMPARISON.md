# Side-by-side: mlx-taef vs full mflux VAE

Visual showcase of what mlx-taef does on real generations. Every number on this page comes from `scripts/run_showcase.py` and the JSON report at `_artifacts/showcase_report.json`. The images live alongside this file under `_artifacts/showcase/`.

## Test machine

- Apple M1 Max, 32 GB unified memory (`sysctl machdep.cpu.brand_string` + `hw.memsize`)
- macOS Darwin 25.6.0, Python 3.13.12
- mflux 0.18.1, mlx-teacache 0.9.3, MLX 0.31.2
- mlx-taef source `v0.7.1-8-g28af6c5` at commit `28af6c5`; installed distribution `0.7.2.dev4+ga4e5df5eb.d20260809`
- The three FLUX.2 scenarios (`taef2_vs_vae`, `live_preview`, `combined`) were re-measured on 2026-09-17 under macOS Darwin 27.0.0, mflux 0.19.1, MLX 0.32.2 and mlx-taef `v0.8.1-5-g5d97cd5`, after the TAEF2 input fix described below. The report records that run under `scenario_updates`; the FLUX.1 and Z-Image scenarios keep the run described above.
- Quantization: int4 (mflux `quantize=4`), bf16 generation, fp32 tiny-autoencoder decode
- Every condition ran in an isolated subprocess with `mx.set_wired_limit` set per the cap column. Each decode was timed after one untimed warmup call, so the figure reflects steady-state per-step decode rather than a cold first call. Every model-loading subprocess — the three live-generation workers and the vs-VAE decode reps alike — enforces a 28 GiB ceiling on MLX active memory (runs after 2026-09-18 count active plus retained cache against the same ceiling and record the reading; the numbers here predate that change); the live-generation workers add a 55-minute wall budget on top. Hardware metadata is recorded inline in `_artifacts/showcase_report.json`.

## Where this fits

- TAEF2's upstream model card markets it as a *real-time previewing* tool for FLUX.2 generation ([huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2)).
- That card explicitly notes: *"Unlike TAEF1, TAEF2's architecture isn't properly integrated into Diffusers yet. So for now you'll want some wrapper code"* — this is the gap mlx-taef fills on the MLX side.
- Upstream is honest about the trade: TAESD *"trades a (modest) loss in quality for a (substantial) gain in speed"* and *"tends to fudge fine details"* ([github.com/madebyollin/taesd](https://github.com/madebyollin/taesd)). The SSIM numbers below land squarely inside that frame.
- HuggingFace Diffusers users get `AutoencoderTiny` plus `callback_on_step_end` ([huggingface.co/docs/diffusers/api/models/autoencoder_tiny](https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny)). mlx-taef's `LivePreviewCallback` is the mflux equivalent.
- ComfyUI users get `--preview-method taesd` decoding every step. mlx-taef's `LivePreviewCallback` defaults to `every=5` (amortizes the ~30 ms TAEF2 decode); pass `every=1` to match ComfyUI's per-step behavior.

## Scenarios

### `taef2_vs_vae` — TAEF2 decoder vs Full FLUX.2 VAE (same latent)

Same FLUX.2 Klein base 4B latent, two different decoders. Both produce a 512×512 RGB image; SSIM and LPIPS are each the cross-product of TAEF2 reps (5) against vanilla VAE reps (3), and all 15 pairs land at the same value because each decoder is deterministic on the same latent. One direction note for the rest of this page: SSIM is a similarity score, so higher is better. LPIPS is a learned perceptual distance, so lower is better.

| | Vanilla FLUX.2 VAE | TAEF2 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.265 s | 0.0304 s |
| Decode latency range | 0.2650 to 0.2657 s | 0.0303 to 0.0308 s |
| Peak decode memory (post-model-load) | 2.80 GB | 0.59 GB |
| Applied wired cap | 12 GB | 2 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef2/vae/vanilla_vae_rep0.webp) | ![taef2](_artifacts/showcase/taef2/taef/taef2_rep0.webp) |

**TAEF2 is ~8.7× faster, with ~4.8× lower peak decode memory.** SSIM(TAEF2, Vanilla) = **0.960**, LPIPS(TAEF2, Vanilla) = **0.057** (15/15 pairs each).

Up to v0.8.1 this row read SSIM 0.616. Those releases applied the Flux2VAE batch-norm inverse to the latent before TAEF2, on the reasoning that the full VAE does the same. TAEF2 turns out to want the normalized latent the diffusion model emits, which is also how its upstream reference code and ComfyUI feed it. The inverse widens the latent by about 1.8× per channel (the VAE's running variance is close to 3), and the decode shows it: shadows crushed, highlights clipped, color oversaturated. `scripts/ab_taef2_bn_domain.py` decodes one latent both ways and scores each against the full VAE, with lossless PNGs between decoder and scorer. Without the inverse: SSIM 0.920, LPIPS 0.058. With it: 0.588 and 0.241. On a fully denoised 768×512 portrait the gap is the same shape, 0.945 / 0.022 against 0.716 / 0.150. The images and reports for both are committed under `_artifacts/ab_taef2_bn_domain/`; the portrait latent is not, and `control_portrait_768x512/RECIPE.md` there gives the capture command. The 0.960 in the table is higher than the A/B's 0.920 because this page's protocol scores webp files, and the codec smooths the same noise out of both images.

TAEF2 is still a 4 MB decoder standing in for a ~340 MB VAE, and it softens fine detail such as specular highlights and micro-texture. If you need final-quality output, use the full VAE: the decode step alone costs about 0.27 s and 2.8 GB, on top of the multi-GB model construction the tiny autoencoder skips entirely.

`scripts/diff_showcase_report.py` locks the floor at `ssim_median - 0.05` (so 0.910 here) to catch regressions.

### `taef1_vs_vae` — TAEF1 decoder vs Full FLUX.1 VAE

Same setup, FLUX.1-dev side. TAEF1 has been around longer and its architecture is closer to the FLUX.1 VAE it shadows.

| | Vanilla FLUX.1 VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.300 s | 0.0295 s |
| Decode latency range | 0.2960 to 0.3091 s | 0.0294 to 0.0309 s |
| Peak decode memory (post-model-load) | 3.67 GB | 0.55 GB |
| Applied wired cap | 6 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef1/vae/vanilla_vae_rep0.webp) | ![taef1](_artifacts/showcase/taef1/taef/taef1_rep0.webp) |

**TAEF1 is ~10.2× faster, with ~6.7× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.939**, LPIPS(TAEF1, Vanilla) = **0.026** (15/15 pairs each).

The taef1 image is nearly indistinguishable from the vanilla FLUX.1 VAE output by eye — the SSIM bears that out. If you're previewing FLUX.1-dev or schnell, TAEF1 is essentially a free win.

### `zimage_vs_vae` — TAEF1 decoder vs Full Z-Image VAE (same latent)

Z-Image-Turbo shares FLUX.1's 16-channel latent contract, so the existing TAEF1 decoder previews it with no new weights. Same setup as the two scenarios above: one Z-Image-Turbo latent, decoded once by TAEF1 and once by the full Z-Image VAE.

| | Vanilla Z-Image VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.237 s | 0.0295 s |
| Decode latency range | 0.2337 to 0.2384 s | 0.0295 to 0.0307 s |
| Peak decode memory (post-model-load) | 2.61 GB | 0.55 GB |
| Applied wired cap | 4 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/zimage/vae/vanilla_vae_rep0.webp) | ![zimage](_artifacts/showcase/zimage/taef/zimage_rep0.webp) |

**TAEF1 is ~8.0× faster on the Z-Image latent, with ~4.8× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.940**, LPIPS(TAEF1, Vanilla) = **0.027** (15/15 pairs each). The same decoder produces the comparable FLUX.1 fidelity above.

### `live_preview` — full FLUX.2 generation with per-step TAEF2 previews

One full FLUX.2 Klein base 4B generation, 4 inference steps, seed=42, prompt "a red apple on a wooden table". `LivePreviewCallback(variant="taef2", numbered_frames=True, every=1)` decodes a TAEF2 preview at every step and saves it as `live_preview_step{NN}.webp`. The final image is decoded by the full FLUX.2 VAE (mflux's native return path) and saved as `live_preview_final.webp`.

- Wall-clock: **10.07 s** total (model load + 4 generation steps + 4 TAEF2 previews + final VAE decode)
- Peak memory: **10.74 GB** (whole-process, includes Flux2Klein + TAEF2 + transformer activations)
- Gallery: `_artifacts/showcase/live_preview/live_preview_step00..03.webp`
- Final: `_artifacts/showcase/live_preview/live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/live_preview/live_preview_step00.webp) | ![s1](_artifacts/showcase/live_preview/live_preview_step01.webp) | ![s2](_artifacts/showcase/live_preview/live_preview_step02.webp) | ![s3](_artifacts/showcase/live_preview/live_preview_step03.webp) | ![final](_artifacts/showcase/live_preview/live_preview_final.webp) |

That's the live-preview loop in practice: noise resolves into a recognizable image, and each step's preview costs a fraction of a full VAE decode.

### `zimage_live_preview` — full Z-Image-Turbo generation with per-step TAEF1 previews

Same recipe on the Z-Image-Turbo side: one full generation, 4 steps, seed=42, "a red apple on a wooden table", with a TAEF1 preview decoded at every step and the final image handed back by mflux's own Z-Image VAE.

- Wall-clock: **25.78 s** total (thermally sensitive; see the note below)
- Peak memory: **25.92 GB** (whole-process; Z-Image-Turbo's transformer is the dominant cost here, not the preview decoder)
- Gallery: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00..03.webp`
- Final: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![z0](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00.webp) | ![z1](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step01.webp) | ![z2](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step02.webp) | ![z3](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step03.webp) | ![zf](_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp) |

The higher wall-clock and peak memory next to `live_preview` come from Z-Image-Turbo's generation cost, not the preview decoder. Its 25.9 GB working set dominates this scenario. The full-generation timing is thermally sensitive, while the isolated TAEF1 decode remains about 30 ms regardless of which model produced the latent.

### `combined` — mflux + TAEF2 previews + mlx-teacache step-skipping

Same generation as `live_preview`, but with `apply_teacache(flux)` wrapping the transformer before the loop runs. TeaCache skips noise-prediction work when the residual is small enough; with the default `skip_first_n_steps=1` and `skip_last_n_steps=1`, only 2 of 4 steps are candidates for skipping in a 4-step run.

- Wall-clock: **7.93 s** total (vs `live_preview`'s 10.07 s, a **1.27× speedup**)
- Peak memory: **5.55 GB** (vs 10.74 GB, **48% less**)
- TeaCache stats: 1 step skipped, 1 step computed, variant=`flux2-klein-base-4b`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/combined/combined_step00.webp) | ![s1](_artifacts/showcase/combined/combined_step01.webp) | ![s2](_artifacts/showcase/combined/combined_step02.webp) | ![s3](_artifacts/showcase/combined/combined_step03.webp) | ![final](_artifacts/showcase/combined/combined_final.webp) |

A few honest notes on this number:

- 1 skip out of 4 is a small sample. The full speedup curve scales with step count — at 28 steps and the same rel-l1 threshold, the skip count is far higher.
- Each wall-clock figure is a single whole-generation run, and it moves with the mflux and MLX versions: the previous capture (mflux 0.18.1, MLX 0.31.2) read 12.71 s against 9.01 s, a 1.41× ratio. Treat the ratio as indicative.
- The 48% peak-memory drop is partly the skipped transformer call (whose activations never materialise) and partly the mflux compiled-path interaction noted in mlx-teacache's own release notes. Be careful attributing it all to one cause.
- The two libraries compose cleanly: mlx-teacache wraps the transformer, mlx-taef hooks the callback registry. Neither knows about the other.

## Reproducing these numbers

```bash
# Step 1: refresh fixture latents (heavy; one-time per variant)
uv run python scripts/_capture_latent.py --variant flux1-dev
uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b
uv run python scripts/_capture_latent.py --variant z-image-turbo

# Step 2: run all 6 scenarios
uv run python scripts/run_showcase.py --scenario all \
    --report _artifacts/showcase_report.json

# Step 3: extract the tagged v0.6.2 report and compare it with v0.7.0.
# The differ migrates the schema-v1 baseline to schema v2 while loading it.
git show v0.6.2:_artifacts/showcase_report.json \
    > /tmp/mlx-taef-v0.6.2-showcase.json
uv run python scripts/diff_showcase_report.py \
    /tmp/mlx-taef-v0.6.2-showcase.json _artifacts/showcase_report.json
```

Wall-time on M1 Max: a few minutes for all 6 scenarios with the three latents already captured — the two full-generation scenarios (`live_preview`, `zimage_live_preview`) dominate the total; the four vs-VAE decode comparisons are each well under a second of actual decode time. Latent capture adds a few more minutes on top.

## Honest-claim discipline

Every number on this page ties to a measurement in the committed JSON at `_artifacts/showcase_report.json`. No hand-waved performance numbers.

**A methodology correction:** earlier releases (v0.2.0 through v0.6.1) reported "decode latency" by timing model construction together with the decode call, which put the tiny decoder at ~180–260 ms — mostly construction, not decode. v0.6.2 measures only the decode step: model construction and latent unpacking run outside the clock, and each decode is timed after one untimed warmup call, so the figure is the steady-state per-step cost a live preview actually pays after its first step, with weights resident and the Metal kernels already compiled. Measured that way, all three decoders land at ~30 ms — as expected for same-size, same-architecture decoders — so the "decoder step *in isolation*" language on this page now describes exactly what the clock captures.

The headline `~8–10×` decode-speedup numbers above are for the decoder step *in isolation*, on the same latent, timed at steady state in separate subprocesses. They measure the decode step alone; the `live_preview` / `zimage_live_preview` / `combined` scenarios show the whole-generation picture users see end to end. The decoder speedup matters most for live previews — every step is a separate decode, and you pay it once per step.

SSIM thresholds: the 0.75 figure was a starting heuristic. The first bench run validates it; the regression check locks the floor at `ssim_median - 0.05`. All three decoders clear it with room to spare (0.94 to 0.96).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
