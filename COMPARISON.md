# Side-by-side: mlx-taef vs full mflux VAE

Visual showcase of what mlx-taef does on real generations. Every number on this page comes from `scripts/run_showcase.py` and the JSON report at `_artifacts/showcase_report.json`. The images are committed alongside this file under `_artifacts/showcase/`.

## Test machine

<!-- TBD after bench run -->

## Ecosystem positioning

- TAEF2's upstream model card markets it as a *real-time previewing* tool for FLUX.2 generation ([huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2)).
- That card explicitly notes: *"Unlike TAEF1, TAEF2's architecture isn't properly integrated into Diffusers yet. So for now you'll want some wrapper code"* — this is the gap mlx-taef fills on the MLX side.
- Upstream is honest about the trade: TAESD *"trades a (modest) loss in quality for a (substantial) gain in speed"* and *"tends to fudge fine details"* ([github.com/madebyollin/taesd](https://github.com/madebyollin/taesd)). The SSIM numbers below are measured under that frame.
- HuggingFace Diffusers users have `AutoencoderTiny` plus `callback_on_step_end` ([huggingface.co/docs/diffusers/api/models/autoencoder_tiny](https://huggingface.co/docs/diffusers/api/models/autoencoder_tiny)). mlx-taef's `LivePreviewCallback` is the mflux equivalent.
- ComfyUI users have `--preview-method taesd` decoding every step. mlx-taef's `LivePreviewCallback` defaults to `every=5` (amortizes the ~100 ms TAEF2 decode); pass `every=1` to match ComfyUI's per-step behavior.

## Scenarios

### `taef2_vs_vae` — TAEF2 decoder vs Full FLUX.2 VAE (same latent)

<!-- TBD after bench run -->

| | Vanilla FLUX.2 VAE | TAEF2 |
|---|---|---|
| Decode latency (median) | `<!-- TBD -->` | `<!-- TBD -->` |
| Peak memory | `<!-- TBD -->` | `<!-- TBD -->` |
| Applied cap | `<!-- TBD -->` | `<!-- TBD -->` |
| Image | `<!-- TBD: _artifacts/showcase/taef2_vs_vae/vanilla.webp -->` | `<!-- TBD: _artifacts/showcase/taef2_vs_vae/taef2.webp -->` |

SSIM(TAEF2, Vanilla): `<!-- TBD -->` (threshold 0.75)

### `taef1_vs_vae` — TAEF1 decoder vs Full FLUX.1 VAE

<!-- TBD after bench run -->

### `live_preview` — gallery during FLUX.2 generation

<!-- TBD after bench run -->

### `combined` — mflux + TAEF live preview + mlx-teacache step-skipping

<!-- TBD after bench run -->

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

Wall-time on M1 Max: ~30–40 min for all 4 scenarios (depends on which variants you re-capture).

## Honest-claim discipline

Every number on this page ties to a measurement in the committed JSON. README's `~100 ms` claim is replaced with the measured median + range. No hand-waved performance numbers in v0.2.0+ docs. SSIM threshold (0.75) is a starting heuristic; the first bench run validates it, and the second-run-forward regression check locks the floor at `ssim_median - 0.05`.
