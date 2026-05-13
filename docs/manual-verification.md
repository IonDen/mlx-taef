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
   from mflux import Flux2Klein
   from mlx_taef.integrations.mflux import LivePreviewCallback

   model = Flux2Klein.from_pretrained("4bit")
   callback = LivePreviewCallback(
       variant="taef2",
       every=5,
       save_to=Path("preview.png"),
       latent_height=32,  # 512 / 16
       latent_width=32,
   )
   model.generate_image(
       prompt="a red apple on a wooden table",
       steps=25,
       width=512,
       height=512,
       seed=42,
       callbacks=[callback],
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
