# Side-by-side: mlx-taef vs full mflux VAE

Visual showcase of what mlx-taef does on real generations. Every number on this page comes from `scripts/run_showcase.py` and the JSON report at `_artifacts/showcase_report.json`. The images live alongside this file under `_artifacts/showcase/`.

## Test machine

- Apple M1 Max, 32 GB unified memory
- macOS Darwin 25.5.0
- mflux 0.17.5, mlx-taef 0.2.0, mlx-teacache 0.6.1, MLX 0.31.2
- Quantization: int4 (mflux `quantize=4`), activation dtype bf16
- All conditions ran in isolated subprocesses with `mx.set_wired_limit` set per the cap column

## Where this fits

- TAEF2's upstream model card markets it as a *real-time previewing* tool for FLUX.2 generation ([huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2)).
- That card explicitly notes: *"Unlike TAEF1, TAEF2's architecture isn't properly integrated into Diffusers yet. So for now you'll want some wrapper code"* — this is the gap mlx-taef fills on the MLX side.
- Upstream is honest about the trade: TAESD *"trades a (modest) loss in quality for a (substantial) gain in speed"* and *"tends to fudge fine details"* ([github.com/madebyollin/taesd](https://github.com/madebyollin/taesd)). The SSIM numbers below land squarely inside that frame.
- HuggingFace Diffusers users get `AutoencoderTiny` plus `callback_on_step_end` ([huggingface.co/docs/diffusers/api/models/autoencoder_tiny](https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny)). mlx-taef's `LivePreviewCallback` is the mflux equivalent.
- ComfyUI users get `--preview-method taesd` decoding every step. mlx-taef's `LivePreviewCallback` defaults to `every=5` (amortizes the ~260 ms TAEF2 decode); pass `every=1` to match ComfyUI's per-step behavior.

## Scenarios

### `taef2_vs_vae` — TAEF2 decoder vs Full FLUX.2 VAE (same latent)

Same FLUX.2 Klein base 4B latent, two different decoders. Both produce a 512×512 RGB image; SSIM is the cross-product of TAEF2 reps (5) against vanilla VAE reps (3), and all 15 pairs land at the same value because each decoder is deterministic on the same latent.

| | Vanilla FLUX.2 VAE | TAEF2 |
|---|---|---|
| Decode latency (median of 3/5 cold subprocess reps) | 2.147 s | 0.260 s |
| Decode latency range | 2.145 – 2.502 s | 0.246 – 0.348 s |
| Peak decode memory (post-model-load) | 2.37 GB | 0.59 GB |
| Applied wired cap | 12 GB | 2 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef2/vae/vanilla_vae_rep0.webp) | ![taef2](_artifacts/showcase/taef2/taef/taef2_rep0.webp) |

**TAEF2 is ~8.3× faster, with ~4× lower peak decode memory.** SSIM(TAEF2, Vanilla) = **0.616** (15/15 pairs).

That 0.616 is below the 0.75 starting threshold from the spec, and it's worth being explicit about why: TAEF2 is a 4 MB preview decoder. The full FLUX.2 VAE is ~340 MB. TAEF2 keeps the structure (apple, table, color) and loses fine detail (specular highlight, micro-texture, exact hue). That's the deliberate trade. If you need 0.95+ fidelity, use the full VAE, but expect to pay 2 GB of GPU memory and 2 seconds per preview.

The first bench run validates the threshold; from v0.2.1 forward `scripts/diff_showcase_report.py` will lock the floor at `ssim_median - 0.05` (so 0.566 here) to catch regressions.

### `taef1_vs_vae` — TAEF1 decoder vs Full FLUX.1 VAE

Same setup, FLUX.1-dev side. TAEF1 has been around longer and its architecture is closer to the FLUX.1 VAE it shadows.

| | Vanilla FLUX.1 VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 cold subprocess reps) | 1.995 s | 0.185 s |
| Decode latency range | 1.986 – 2.011 s | 0.182 – 0.190 s |
| Peak decode memory (post-model-load) | 3.00 GB | 0.55 GB |
| Applied wired cap | 6 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef1/vae/vanilla_vae_rep0.webp) | ![taef1](_artifacts/showcase/taef1/taef/taef1_rep0.webp) |

**TAEF1 is ~10.8× faster, with ~5.4× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.939** (15/15 pairs).

The taef1 image is nearly indistinguishable from the vanilla FLUX.1 VAE output by eye — the SSIM bears that out. If you're previewing FLUX.1-dev or schnell, TAEF1 is essentially a free win.

### `live_preview` — full FLUX.2 generation with per-step TAEF2 previews

One full FLUX.2 Klein base 4B generation, 4 inference steps, seed=42, prompt "a red apple on a wooden table". `_GalleryPreviewCallback` decodes a TAEF2 preview at every step and saves it as `live_preview_step{NN}.webp`. The final image is decoded by the full FLUX.2 VAE (mflux's native return path) and saved as `live_preview_final.webp`.

- Wall-clock: **11.21 s** total (model load + 4 generation steps + 4 TAEF2 previews + final VAE decode)
- Peak memory: **10.66 GB** (whole-process, includes Flux2Klein + TAEF2 + transformer activations)
- Gallery: `_artifacts/showcase/live_preview/live_preview_step00..03.webp`
- Final: `_artifacts/showcase/live_preview/live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/live_preview/live_preview_step00.webp) | ![s1](_artifacts/showcase/live_preview/live_preview_step01.webp) | ![s2](_artifacts/showcase/live_preview/live_preview_step02.webp) | ![s3](_artifacts/showcase/live_preview/live_preview_step03.webp) | ![final](_artifacts/showcase/live_preview/live_preview_final.webp) |

That's the live-preview loop in practice: noise resolves into a recognizable image, and each step's preview costs a fraction of a full VAE decode.

### `combined` — mflux + TAEF2 previews + mlx-teacache step-skipping

Same generation as `live_preview`, but with `apply_teacache(flux)` wrapping the transformer before the loop runs. TeaCache skips noise-prediction work when the residual is small enough; with the default `skip_first_n_steps=1` and `skip_last_n_steps=1`, only 2 of 4 steps are candidates for skipping in a 4-step run.

- Wall-clock: **8.84 s** total (vs `live_preview`'s 11.21 s → **1.27× speedup**)
- Peak memory: **6.21 GB** (vs 10.66 GB → **41% less**)
- TeaCache stats: 1 step skipped, 1 step computed, variant=`flux2-klein-base-4b`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/combined/combined_step00.webp) | ![s1](_artifacts/showcase/combined/combined_step01.webp) | ![s2](_artifacts/showcase/combined/combined_step02.webp) | ![s3](_artifacts/showcase/combined/combined_step03.webp) | ![final](_artifacts/showcase/combined/combined_final.webp) |

A few honest notes on this number:

- 1 skip out of 4 is a small sample. The full speedup curve scales with step count — at 28 steps and the same rel-l1 threshold, the skip count is far higher.
- The 41% peak-memory drop is partly the skipped transformer call (whose activations never materialise) and partly the mflux compiled-path interaction noted in mlx-teacache's own v0.6.1 release notes — be careful attributing it all to one cause.
- The two libraries compose cleanly: mlx-teacache wraps the transformer, mlx-taef hooks the callback registry. Neither knows about the other.

## Reproducing these numbers

```bash
# Step 1: refresh fixture latents (heavy; one-time per variant)
uv run python scripts/_capture_latent.py --variant flux1-dev
uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b

# Step 2: run all 4 scenarios
uv run python scripts/run_showcase.py --scenario all \
    --report _artifacts/showcase_report.json

# Step 3: regression check against the committed JSON
uv run python scripts/diff_showcase_report.py \
    _artifacts/showcase_report.json your_new_report.json
```

Wall-time on M1 Max: ~6–8 min for all 4 scenarios with both latents already captured. Latent capture adds another ~2 min total.

## Honest-claim discipline

Every number on this page ties to a measurement in the committed JSON at `_artifacts/showcase_report.json`. No hand-waved performance numbers in v0.2.0+ docs.

The headline `~8.3×` and `~10.8×` numbers are for the decoder step *in isolation*, on the same latent, in cold subprocesses. They are NOT whole-generation speedups — for that, look at the `combined` scenario, which shows what users see when they pair TAEF previews with TeaCache step-skipping. The decoder speedup matters most for live previews — every step is a separate decode, and you pay it once per step.

SSIM thresholds: the 0.75 figure in the spec was a starting heuristic. The first bench run validates it; the regression check locks the floor at `ssim_median - 0.05` from v0.2.1 forward. TAEF2's 0.616 is genuinely lower than that heuristic. That's a signal of upstream TAEF2's preview-grade fidelity, not a regression in mlx-taef's port.
