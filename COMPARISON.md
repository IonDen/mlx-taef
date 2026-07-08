# Side-by-side: mlx-taef vs full mflux VAE

Visual showcase of what mlx-taef does on real generations. Every number on this page comes from `scripts/run_showcase.py` and the JSON report at `_artifacts/showcase_report.json`. The images live alongside this file under `_artifacts/showcase/`.

## Test machine

- Apple M1 Max, 32 GB unified memory (`sysctl machdep.cpu.brand_string` + `hw.memsize`)
- macOS Darwin 25.5.0, Python 3.14.5
- mflux 0.18.0, mlx-teacache 0.9.1, MLX 0.31.2
- mlx-taef at commit `d0cd59a` (the pre-tag dev build that produced the committed report; v0.6.2 re-measures these with decode-in-isolation timing)
- Quantization: int4 (mflux `quantize=4`), activation dtype bf16
- All conditions ran in isolated subprocesses with `mx.set_wired_limit` set per the cap column. Hardware metadata is recorded inline in `_artifacts/showcase_report.json` for auditability.

## Where this fits

- TAEF2's upstream model card markets it as a *real-time previewing* tool for FLUX.2 generation ([huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2)).
- That card explicitly notes: *"Unlike TAEF1, TAEF2's architecture isn't properly integrated into Diffusers yet. So for now you'll want some wrapper code"* — this is the gap mlx-taef fills on the MLX side.
- Upstream is honest about the trade: TAESD *"trades a (modest) loss in quality for a (substantial) gain in speed"* and *"tends to fudge fine details"* ([github.com/madebyollin/taesd](https://github.com/madebyollin/taesd)). The SSIM numbers below land squarely inside that frame.
- HuggingFace Diffusers users get `AutoencoderTiny` plus `callback_on_step_end` ([huggingface.co/docs/diffusers/api/models/autoencoder_tiny](https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny)). mlx-taef's `LivePreviewCallback` is the mflux equivalent.
- ComfyUI users get `--preview-method taesd` decoding every step. mlx-taef's `LivePreviewCallback` defaults to `every=5` (amortizes the ~45 ms TAEF2 decode); pass `every=1` to match ComfyUI's per-step behavior.

## Scenarios

### `taef2_vs_vae` — TAEF2 decoder vs Full FLUX.2 VAE (same latent)

Same FLUX.2 Klein base 4B latent, two different decoders. Both produce a 512×512 RGB image; SSIM is the cross-product of TAEF2 reps (5) against vanilla VAE reps (3), and all 15 pairs land at the same value because each decoder is deterministic on the same latent.

| | Vanilla FLUX.2 VAE | TAEF2 |
|---|---|---|
| Decode latency (median of 3/5 cold subprocess reps) | 0.329 s | 0.046 s |
| Decode latency range | 0.329 – 0.332 s | 0.044 – 0.048 s |
| Peak decode memory (post-model-load) | 2.57 GB | 0.55 GB |
| Applied wired cap | 12 GB | 2 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef2/vae/vanilla_vae_rep0.webp) | ![taef2](_artifacts/showcase/taef2/taef/taef2_rep0.webp) |

**TAEF2 is ~7.2× faster, with ~4.7× lower peak decode memory.** SSIM(TAEF2, Vanilla) = **0.616** (15/15 pairs).

That 0.616 is below the 0.75 starting threshold, and it's worth being explicit about why: TAEF2 is a 4 MB preview decoder. The full FLUX.2 VAE is ~340 MB. TAEF2 keeps the structure (apple, table, color) and loses fine detail (specular highlight, micro-texture, exact hue). That's the deliberate trade. If you need 0.95+ fidelity, use the full VAE — the decode step alone costs about 0.33 s and 2.6 GB, on top of the multi-GB model construction the tiny autoencoder skips entirely.

The first bench run validates the threshold; `scripts/diff_showcase_report.py` locks the floor at `ssim_median - 0.05` (so 0.566 here) to catch regressions.

### `taef1_vs_vae` — TAEF1 decoder vs Full FLUX.1 VAE

Same setup, FLUX.1-dev side. TAEF1 has been around longer and its architecture is closer to the FLUX.1 VAE it shadows.

| | Vanilla FLUX.1 VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 cold subprocess reps) | 0.352 s | 0.045 s |
| Decode latency range | 0.352 – 0.354 s | 0.042 – 0.046 s |
| Peak decode memory (post-model-load) | 3.08 GB | 0.52 GB |
| Applied wired cap | 6 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef1/vae/vanilla_vae_rep0.webp) | ![taef1](_artifacts/showcase/taef1/taef/taef1_rep0.webp) |

**TAEF1 is ~7.9× faster, with ~5.9× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.939** (15/15 pairs).

The taef1 image is nearly indistinguishable from the vanilla FLUX.1 VAE output by eye — the SSIM bears that out. If you're previewing FLUX.1-dev or schnell, TAEF1 is essentially a free win.

### `zimage_vs_vae` — TAEF1 decoder vs Full Z-Image VAE (same latent)

Z-Image-Turbo shares FLUX.1's 16-channel latent contract, so the existing TAEF1 decoder previews it with no new weights. Same setup as the two scenarios above: one Z-Image-Turbo latent, decoded once by TAEF1 and once by the full Z-Image VAE.

| | Vanilla Z-Image VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 cold subprocess reps) | 0.286 s | 0.058 s |
| Decode latency range | 0.284 – 0.290 s | 0.057 – 0.060 s |
| Peak decode memory (post-model-load) | 1.96 GB | 0.55 GB |
| Applied wired cap | 4 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/zimage/vae/vanilla_vae_rep0.webp) | ![zimage](_artifacts/showcase/zimage/taef/zimage_rep0.webp) |

**TAEF1 is ~5.0× faster on the Z-Image latent, with ~3.6× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.940** (15/15 pairs) — consistent with the FLUX.1 result above, since it's the same decoder.

### `live_preview` — full FLUX.2 generation with per-step TAEF2 previews

One full FLUX.2 Klein base 4B generation, 4 inference steps, seed=42, prompt "a red apple on a wooden table". `LivePreviewCallback(flux=model, numbered_frames=True, every=1)` decodes a TAEF2 preview at every step and saves it as `live_preview_step{NN}.webp`. The final image is decoded by the full FLUX.2 VAE (mflux's native return path) and saved as `live_preview_final.webp`.

- Wall-clock: **11.51 s** total (model load + 4 generation steps + 4 TAEF2 previews + final VAE decode)
- Peak memory: **10.66 GB** (whole-process, includes Flux2Klein + TAEF2 + transformer activations)
- Gallery: `_artifacts/showcase/live_preview/live_preview_step00..03.webp`
- Final: `_artifacts/showcase/live_preview/live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/live_preview/live_preview_step00.webp) | ![s1](_artifacts/showcase/live_preview/live_preview_step01.webp) | ![s2](_artifacts/showcase/live_preview/live_preview_step02.webp) | ![s3](_artifacts/showcase/live_preview/live_preview_step03.webp) | ![final](_artifacts/showcase/live_preview/live_preview_final.webp) |

That's the live-preview loop in practice: noise resolves into a recognizable image, and each step's preview costs a fraction of a full VAE decode.

### `zimage_live_preview` — full Z-Image-Turbo generation with per-step TAEF1 previews

Same recipe on the Z-Image-Turbo side: one full generation, 4 steps, seed=42, "a red apple on a wooden table", with a TAEF1 preview decoded at every step and the final image handed back by mflux's own Z-Image VAE.

- Wall-clock: **36.84 s** total
- Peak memory: **25.92 GB** (whole-process; Z-Image-Turbo's transformer is the dominant cost here, not the preview decoder)
- Gallery: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00..03.webp`
- Final: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![z0](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00.webp) | ![z1](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step01.webp) | ![z2](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step02.webp) | ![z3](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step03.webp) | ![zf](_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp) |

The much higher wall-clock and peak memory next to `live_preview` come from Z-Image-Turbo's own generation cost, not the preview decoder. TAEF1's per-step decode is still the same ~45–60 ms measured above, regardless of which model produced the latent.

### `combined` — mflux + TAEF2 previews + mlx-teacache step-skipping

Same generation as `live_preview`, but with `apply_teacache(flux)` wrapping the transformer before the loop runs. TeaCache skips noise-prediction work when the residual is small enough; with the default `skip_first_n_steps=1` and `skip_last_n_steps=1`, only 2 of 4 steps are candidates for skipping in a 4-step run.

- Wall-clock: **8.83 s** total (vs `live_preview`'s 11.51 s → **1.30× speedup**)
- Peak memory: **5.55 GB** (vs 10.66 GB → **48% less**)
- TeaCache stats: 1 step skipped, 1 step computed, variant=`flux2-klein-base-4b`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/combined/combined_step00.webp) | ![s1](_artifacts/showcase/combined/combined_step01.webp) | ![s2](_artifacts/showcase/combined/combined_step02.webp) | ![s3](_artifacts/showcase/combined/combined_step03.webp) | ![final](_artifacts/showcase/combined/combined_final.webp) |

A few honest notes on this number:

- 1 skip out of 4 is a small sample. The full speedup curve scales with step count — at 28 steps and the same rel-l1 threshold, the skip count is far higher.
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

# Step 3: regression check against the committed JSON
uv run python scripts/diff_showcase_report.py \
    _artifacts/showcase_report.json your_new_report.json
```

Wall-time on M1 Max: a few minutes for all 6 scenarios with the three latents already captured — the two full-generation scenarios (`live_preview`, `zimage_live_preview`) dominate the total; the four vs-VAE decode comparisons are each well under a second of actual decode time. Latent capture adds a few more minutes on top.

## Honest-claim discipline

Every number on this page ties to a measurement in the committed JSON at `_artifacts/showcase_report.json`. No hand-waved performance numbers.

**A methodology correction:** earlier releases (v0.2.0 through v0.6.1) measured "decode latency" by timing model construction and the decode call together, in one window. That inflated both the absolute times and the speedup ratios — the published ~11.0× (TAEF1) and ~8.3× (TAEF2) figures were mostly model-construction cost, not decode cost. v0.6.2 fixes this: the timed region now covers only the decode step, with model construction and latent unpacking running outside the clock. So the "decoder step *in isolation*" language on this page now describes exactly what the clock captures. It also explains why the old figures diverged: TAEF1 and TAEF2 decode in almost exactly the same time (~45 ms and ~46 ms), as expected for two same-size, same-architecture decoders. The old gap came from construction overhead alone.

The headline `~7–8×` decode-speedup numbers above are for the decoder step *in isolation*, on the same latent, in cold subprocesses. They measure the decode step alone; the `live_preview` / `zimage_live_preview` / `combined` scenarios show the whole-generation picture users see end to end. The decoder speedup matters most for live previews — every step is a separate decode, and you pay it once per step.

SSIM thresholds: the 0.75 figure was a starting heuristic. The first bench run validates it; the regression check locks the floor at `ssim_median - 0.05`. TAEF2's 0.616 is genuinely lower than that heuristic. That's a signal of upstream TAEF2's preview-grade fidelity, not a regression in mlx-taef's port.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
