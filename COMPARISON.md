# Side-by-side: mlx-taef vs full mflux VAE

Visual showcase of what mlx-taef does on real generations. Every number on this page comes from `scripts/run_showcase.py` and the JSON report at `_artifacts/showcase_report.json`. The images live alongside this file under `_artifacts/showcase/`.

## Test machine

- Apple M1 Max, 32 GB unified memory (`sysctl machdep.cpu.brand_string` + `hw.memsize`)
- macOS Darwin 25.5.0, Python 3.14.5
- mflux 0.18.0, mlx-teacache 0.9.1, MLX 0.31.2
- mlx-taef source `v0.6.2-4-g1e79c29` at commit `1e79c29`; installed distribution `0.6.3.dev4+g1e79c2977.d20260724`
- Quantization: int4 (mflux `quantize=4`), bf16 generation, fp32 tiny-autoencoder decode
- Every condition ran in an isolated subprocess with `mx.set_wired_limit` set per the cap column. Each decode was timed after one untimed warmup call, so the figure reflects steady-state per-step decode rather than a cold first call. Live-generation workers also enforced a 28 GiB active-memory ceiling and a 55-minute wall budget. Hardware metadata is recorded inline in `_artifacts/showcase_report.json`.

## Where this fits

- TAEF2's upstream model card markets it as a *real-time previewing* tool for FLUX.2 generation ([huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2)).
- That card explicitly notes: *"Unlike TAEF1, TAEF2's architecture isn't properly integrated into Diffusers yet. So for now you'll want some wrapper code"* — this is the gap mlx-taef fills on the MLX side.
- Upstream is honest about the trade: TAESD *"trades a (modest) loss in quality for a (substantial) gain in speed"* and *"tends to fudge fine details"* ([github.com/madebyollin/taesd](https://github.com/madebyollin/taesd)). The SSIM numbers below land squarely inside that frame.
- HuggingFace Diffusers users get `AutoencoderTiny` plus `callback_on_step_end` ([huggingface.co/docs/diffusers/api/models/autoencoder_tiny](https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny)). mlx-taef's `LivePreviewCallback` is the mflux equivalent.
- ComfyUI users get `--preview-method taesd` decoding every step. mlx-taef's `LivePreviewCallback` defaults to `every=5` (amortizes the ~30 ms TAEF2 decode); pass `every=1` to match ComfyUI's per-step behavior.

## Scenarios

### `taef2_vs_vae` — TAEF2 decoder vs Full FLUX.2 VAE (same latent)

Same FLUX.2 Klein base 4B latent, two different decoders. Both produce a 512×512 RGB image; SSIM is the cross-product of TAEF2 reps (5) against vanilla VAE reps (3), and all 15 pairs land at the same value because each decoder is deterministic on the same latent.

| | Vanilla FLUX.2 VAE | TAEF2 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.283 s | 0.0305 s |
| Decode latency range | 0.2832 to 0.2834 s | 0.0304 to 0.0320 s |
| Peak decode memory (post-model-load) | 2.80 GB | 0.59 GB |
| Applied wired cap | 12 GB | 2 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef2/vae/vanilla_vae_rep0.webp) | ![taef2](_artifacts/showcase/taef2/taef/taef2_rep0.webp) |

**TAEF2 is ~9.3× faster, with ~4.8× lower peak decode memory.** SSIM(TAEF2, Vanilla) = **0.616** (15/15 pairs).

That 0.616 is below the 0.75 starting threshold, and it's worth being explicit about why: TAEF2 is a 4 MB preview decoder. The full FLUX.2 VAE is ~340 MB. TAEF2 keeps the structure (apple, table, color) and loses fine detail (specular highlight, micro-texture, exact hue). That's the deliberate trade. If you need 0.95+ fidelity, use the full VAE — the decode step alone costs about 0.28 s and 2.8 GB, on top of the multi-GB model construction the tiny autoencoder skips entirely.

The first bench run validates the threshold; `scripts/diff_showcase_report.py` locks the floor at `ssim_median - 0.05` (so 0.566 here) to catch regressions.

### `taef1_vs_vae` — TAEF1 decoder vs Full FLUX.1 VAE

Same setup, FLUX.1-dev side. TAEF1 has been around longer and its architecture is closer to the FLUX.1 VAE it shadows.

| | Vanilla FLUX.1 VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.302 s | 0.0301 s |
| Decode latency range | 0.3016 to 0.3027 s | 0.0296 to 0.0309 s |
| Peak decode memory (post-model-load) | 3.70 GB | 0.55 GB |
| Applied wired cap | 6 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/taef1/vae/vanilla_vae_rep0.webp) | ![taef1](_artifacts/showcase/taef1/taef/taef1_rep0.webp) |

**TAEF1 is ~10.0× faster, with ~6.8× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.939** (15/15 pairs).

The taef1 image is nearly indistinguishable from the vanilla FLUX.1 VAE output by eye — the SSIM bears that out. If you're previewing FLUX.1-dev or schnell, TAEF1 is essentially a free win.

### `zimage_vs_vae` — TAEF1 decoder vs Full Z-Image VAE (same latent)

Z-Image-Turbo shares FLUX.1's 16-channel latent contract, so the existing TAEF1 decoder previews it with no new weights. Same setup as the two scenarios above: one Z-Image-Turbo latent, decoded once by TAEF1 and once by the full Z-Image VAE.

| | Vanilla Z-Image VAE | TAEF1 |
|---|---|---|
| Decode latency (median of 3/5 warmed subprocess reps) | 0.236 s | 0.0297 s |
| Decode latency range | 0.233 to 0.237 s | 0.0294 to 0.0309 s |
| Peak decode memory (post-model-load) | 2.61 GB | 0.55 GB |
| Applied wired cap | 4 GB | 1 GB |
| Reference image | ![vanilla](_artifacts/showcase/zimage/vae/vanilla_vae_rep0.webp) | ![zimage](_artifacts/showcase/zimage/taef/zimage_rep0.webp) |

**TAEF1 is ~8.0× faster on the Z-Image latent, with ~4.8× lower peak decode memory.** SSIM(TAEF1, Vanilla) = **0.940** (15/15 pairs). The same decoder produces the comparable FLUX.1 fidelity above.

### `live_preview` — full FLUX.2 generation with per-step TAEF2 previews

One full FLUX.2 Klein base 4B generation, 4 inference steps, seed=42, prompt "a red apple on a wooden table". `LivePreviewCallback(flux=model, numbered_frames=True, every=1)` decodes a TAEF2 preview at every step and saves it as `live_preview_step{NN}.webp`. The final image is decoded by the full FLUX.2 VAE (mflux's native return path) and saved as `live_preview_final.webp`.

- Wall-clock: **11.18 s** total (model load + 4 generation steps + 4 TAEF2 previews + final VAE decode)
- Peak memory: **10.73 GB** (whole-process, includes Flux2Klein + TAEF2 + transformer activations)
- Gallery: `_artifacts/showcase/live_preview/live_preview_step00..03.webp`
- Final: `_artifacts/showcase/live_preview/live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/live_preview/live_preview_step00.webp) | ![s1](_artifacts/showcase/live_preview/live_preview_step01.webp) | ![s2](_artifacts/showcase/live_preview/live_preview_step02.webp) | ![s3](_artifacts/showcase/live_preview/live_preview_step03.webp) | ![final](_artifacts/showcase/live_preview/live_preview_final.webp) |

That's the live-preview loop in practice: noise resolves into a recognizable image, and each step's preview costs a fraction of a full VAE decode.

### `zimage_live_preview` — full Z-Image-Turbo generation with per-step TAEF1 previews

Same recipe on the Z-Image-Turbo side: one full generation, 4 steps, seed=42, "a red apple on a wooden table", with a TAEF1 preview decoded at every step and the final image handed back by mflux's own Z-Image VAE.

- Wall-clock: **25.11 s** total (thermally sensitive; see the note below)
- Peak memory: **25.92 GB** (whole-process; Z-Image-Turbo's transformer is the dominant cost here, not the preview decoder)
- Gallery: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00..03.webp`
- Final: `_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![z0](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step00.webp) | ![z1](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step01.webp) | ![z2](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step02.webp) | ![z3](_artifacts/showcase/zimage_live_preview/zimage_live_preview_step03.webp) | ![zf](_artifacts/showcase/zimage_live_preview/zimage_live_preview_final.webp) |

The higher wall-clock and peak memory next to `live_preview` come from Z-Image-Turbo's generation cost, not the preview decoder. Its 25.9 GB working set dominates this scenario. The full-generation timing is thermally sensitive, while the isolated TAEF1 decode remains about 30 ms regardless of which model produced the latent.

### `combined` — mflux + TAEF2 previews + mlx-teacache step-skipping

Same generation as `live_preview`, but with `apply_teacache(flux)` wrapping the transformer before the loop runs. TeaCache skips noise-prediction work when the residual is small enough; with the default `skip_first_n_steps=1` and `skip_last_n_steps=1`, only 2 of 4 steps are candidates for skipping in a 4-step run.

- Wall-clock: **8.54 s** total (vs `live_preview`'s 11.18 s, a **1.31× speedup**)
- Peak memory: **6.15 GB** (vs 10.73 GB, **43% less**)
- TeaCache stats: 1 step skipped, 1 step computed, variant=`flux2-klein-base-4b`

| step 00 | step 01 | step 02 | step 03 | final (full VAE) |
|---|---|---|---|---|
| ![s0](_artifacts/showcase/combined/combined_step00.webp) | ![s1](_artifacts/showcase/combined/combined_step01.webp) | ![s2](_artifacts/showcase/combined/combined_step02.webp) | ![s3](_artifacts/showcase/combined/combined_step03.webp) | ![final](_artifacts/showcase/combined/combined_final.webp) |

A few honest notes on this number:

- 1 skip out of 4 is a small sample. The full speedup curve scales with step count — at 28 steps and the same rel-l1 threshold, the skip count is far higher.
- The 44% peak-memory drop is partly the skipped transformer call (whose activations never materialise) and partly the mflux compiled-path interaction noted in mlx-teacache's own release notes. Be careful attributing it all to one cause.
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

SSIM thresholds: the 0.75 figure was a starting heuristic. The first bench run validates it; the regression check locks the floor at `ssim_median - 0.05`. TAEF2's 0.616 is genuinely lower than that heuristic. That's a signal of upstream TAEF2's preview-grade fidelity, not a regression in mlx-taef's port.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
