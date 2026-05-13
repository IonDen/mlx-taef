# mflux FLUX.2 Klein Latent Layout (probed 2026-05-12)

Source: filipstrand/mflux @ `da36fe5e93c761fa7735def46844d05baaa5da2b`

## In-loop latent shape (passed to InLoopCallback.call_in_loop)

Shape: `(B, latent_height * latent_width, 128)` — 3-D packed DiT format.

- `latent_height = image_height // 16`
- `latent_width  = image_width  // 16`

Example for a 1024×1024 image:
- `latent_height = latent_width = 64`
- packed shape: `(1, 4096, 128)`

The 128 channels come from patchifying a 32-channel latent with 2×2 patches:
`32 channels × 2 × 2 = 128 channels` at half spatial resolution.

dtype: `ModelConfig.precision` (bfloat16 by default in mflux).

Value range: **BN-normalized** — zero-mean, unit-variance, using the Flux2VAE's
BatchNorm running statistics (`bn.running_mean`, `bn.running_var`). This is NOT
raw TAEF2-latent space; it must be de-normalized before feeding into TAEF2.

## Unpack formula

From `mflux/models/flux2/variants/txt2img/flux2_klein.py:117–118`:

```python
# latents: (B, lH*lW, 128)  — DiT packed format
packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)
# packed_latents: (B, 128, lH, lW)  — NCHW
decoded = self.vae.decode_packed_latents(packed_latents)
```

`decode_packed_latents` then in `mflux/models/flux2/model/flux2_vae/vae.py:43–50`:

```python
def decode_packed_latents(self, packed_latents: mx.array) -> mx.array:
    bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
    bn_std  = mx.sqrt(self.bn.running_var.reshape(1, -1, 1, 1) + self.bn.eps)
    latents = packed_latents * bn_std + bn_mean      # de-normalize: BN -> raw 128-ch
    latents = self._unpatchify_latents(latents)      # (B, 128, lH, lW) -> (B, 32, lH*2, lW*2)
    return self.decode(latents)                      # standard VAE decode
```

`_unpatchify_latents` in `vae.py:52–58`:

```python
@staticmethod
def _unpatchify_latents(latents: mx.array) -> mx.array:
    batch_size, num_channels, height, width = latents.shape  # (B, 128, lH, lW)
    latents = mx.reshape(latents, (batch_size, num_channels // 4, 2, 2, height, width))
    # -> (B, 32, 2, 2, lH, lW)
    latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))
    # -> (B, 32, lH, 2, lW, 2)
    latents = mx.reshape(latents, (batch_size, num_channels // 4, height * 2, width * 2))
    # -> (B, 32, lH*2, lW*2)  NCHW, raw latent space
    return latents
```

## Pre-decode transform

Before calling `_unpatchify_latents`, mflux de-normalizes the 128-channel packed
latent using the VAE's stored BatchNorm statistics:

```python
latents = packed_latents * bn_std + bn_mean
```

where:
- `bn_mean` comes from `Flux2VAE.bn.running_mean` (shape 128, representing 4×32 channels)
- `bn_std`  comes from `sqrt(Flux2VAE.bn.running_var + bn.eps)`

After de-normalization and unpatchify, the latents are in the same space as
`Flux2VAE.encode()` outputs — i.e. `(mean - shift_factor) * scaling_factor`.
For FLUX.2 Klein, `scaling_factor = 1.0` and `shift_factor = 0.0`, so the
encoded latents are **identical to the raw mean** from the encoder.
This corresponds to TAEF2's raw latent space (values roughly in [-3, 3]).

## TAEF2 compatibility

TAEF2 (`mlx-taef`) expects NHWC latents of shape `(B, H, W, 32)` where
`H = image_height // 8` and `W = image_width // 8`.

The unpatchify step doubles the spatial dimensions:
- `lH = image_height // 16` (in-loop) → `lH * 2 = image_height // 8` (after unpatchify)

To convert an in-loop mflux latent for TAEF2:

1. Reshape + transpose to NCHW 128-channel: `(B, lH*lW, 128)` → `(B, 128, lH, lW)`
2. De-normalize (requires access to `Flux2VAE.bn` stats) or skip if the stats are
   identity (running_mean=0, running_var=1 after initialization — must verify at
   runtime with an actual model checkpoint).
3. Unpatchify: `(B, 128, lH, lW)` → `(B, 32, lH*2, lW*2)` NCHW
4. Transpose to NHWC: `(B, 32, lH*2, lW*2)` → `(B, lH*2, lW*2, 32)`

For a spike / synthetic test, **without the BN stats**, the unpatchify+transpose
alone gives the correct shape and a plausible latent (just not de-normalized).

**If you need exact value fidelity** for a real preview, you must either:
- Pass in the `Flux2VAE.bn` stats at call time, or
- Use `Flux2VAE.decode_packed_latents` directly (which already does BN de-norm
  + unpatchify + decode).

For Phase 3 Task 3.3 the recommended approach is to expose an optional
`bn_stats` argument and default to identity (mean=0, var=1) when not provided.

## Source references

- `mflux/models/flux2/variants/txt2img/flux2_klein.py:117–118` — line where unpack happens
- `mflux/models/flux2/model/flux2_vae/vae.py:43–58` — `decode_packed_latents` + `_unpatchify_latents`
- `mflux/models/flux2/latent_creator/flux2_latent_creator.py:20–22` — `pack_latents`
- `mflux/models/flux2/latent_creator/flux2_latent_creator.py:60–77` — `prepare_latents` (shape math)
- `mflux/callbacks/callback.py:22–32` — `InLoopCallback.call_in_loop` signature
