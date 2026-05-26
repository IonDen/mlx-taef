# Manual verification: mflux LivePreviewCallback

These steps require:
- mflux installed (`pip install mflux`)
- FLUX.2 Klein 4-bit weights downloaded (~5 GB)
- M-series Mac with 16 GB+ unified memory

This validates that `LivePreviewCallback` produces recognizable previews during a real generation. Automated tests use a fake-shaped latent and can't catch value-space drift.

## Steps

1. Install mflux (in the mlx-taef venv):
   ```bash
   uv pip install mflux
   ```

2. Run a generation with the callback:
   ```python
   from pathlib import Path
   from mflux.models.flux2 import Flux2Klein
   from mlx_taef.integrations.mflux import LivePreviewCallback

   model = Flux2Klein.from_pretrained("4bit")
   callback = LivePreviewCallback(
       variant="taef2",
       every=5,
       save_to=Path("preview.png"),
       latent_height=32,  # 512 / 16
       latent_width=32,
   )
   model.callbacks.register(callback)
   model.generate_image(
       prompt="a red apple on a wooden table",
       num_inference_steps=25,
       width=512,
       height=512,
       seed=42,
   )
   ```

3. Verify `preview.png` updates every 5 steps with progressively clearer images. The final preview should be recognizable as "red apple on a wooden table."

## Known limitation

Without passing `bn_mean` and `bn_var` from `Flux2VAE.bn.running_{mean,var}`, the preview uses identity BN — structure is correct but colors may shift. For exact previews:

```python
flux2_vae = model.vae
callback = LivePreviewCallback(
    ...,
    bn_mean=flux2_vae.bn.running_mean,
    bn_var=flux2_vae.bn.running_var,
)
```

## Cross-process MLX non-determinism

mlx-taef output is reproducible *within a single Python process* with a fixed seed, fixed MLX version, and fixed mflux version. **Bit-exact reproducibility across saves/loads/processes is not guaranteed.**

MLX evaluates lazily and walks backward from requested outputs (DFS via `inputs`, then a width-limited BFS execution tape). Holding extra references to intermediate arrays — even via an unevaluated side-array — prevents in-place buffer donation, shifts memory aliases, and changes temporary lifetimes. Combined with shape/device-dependent kernel dispatch in matmul, SDPA, and normalization layers, the resulting floating-point summation order differs slightly across runs. Over many decoder layers this compounds.

**What this means for verification:**

- Don't gate correctness on `mx.array_equal` against a reference *file*. Use a measured numerical tolerance — start at `np.testing.assert_allclose(np.asarray(out), reference, atol=1e-5, rtol=1e-5)` for fp32 (looser for bf16 / quantized).
- Within a single process, `mx.array_equal` between two functionally-equivalent code paths is still useful (e.g., "wrapper at no-op equals inner module called directly").
- Across saves/loads/processes, it isn't.
- This is why `scripts/run_showcase.py` computes SSIM in the orchestrator after both candidate and reference webps are saved to disk: cross-process MLX output is not bit-stable, so SSIM is computed as a measured tolerance metric, not bit-exact equality.

See [`user-mlx-developer` skill gotcha #24](https://github.com/IonDen/dotclaude/blob/main/skills/user-mlx-developer/references/gotchas.md) for the full upstream investigation.

**If you genuinely need bit-exact across runs** (release regression testing, scientific reproducibility), use [`mlx-deterministic`](https://github.com/ProbioticFarmer/mlx-deterministic), which provides batch-invariant Metal kernels at 7–31% overhead. Default to "no" unless you've felt the pain of false-positive parity failures.
