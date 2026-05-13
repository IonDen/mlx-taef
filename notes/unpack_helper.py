"""Phase 0 spike: unpack mflux's packed FLUX.2 latent into NHWC for TAEF2.

Verified against mflux source at commit da36fe5e93c761fa7735def46844d05baaa5da2b.
Move to src/mlx_taef/integrations/mflux.py during Phase 3 Task 3.3.

Pipeline (derived from mflux source inspection):
  in-loop latent: (B, lH*lW, 128)   packed DiT format, BN-normalized
      ↓ reshape + transpose
  (B, 128, lH, lW)  NCHW
      ↓ optional BN de-normalize: latents * bn_std + bn_mean
  (B, 128, lH, lW)  raw 128-ch patchified latent
      ↓ _unpatchify: reshape(B, 32, 2, 2, lH, lW) → transpose(0,1,4,2,5,3) → reshape(B, 32, lH*2, lW*2)
  (B, 32, lH*2, lW*2)  NCHW, 32-ch raw latent (image_height//8 × image_width//8)
      ↓ transpose(0, 2, 3, 1)
  (B, lH*2, lW*2, 32)  NHWC  ← TAEF2.decode() input

See notes/mflux-latent-layout.md for the detailed analysis.
"""

from __future__ import annotations

import mlx.core as mx


def unpack_flux2_latent(
    packed: mx.array,
    *,
    latent_height: int,
    latent_width: int,
    bn_mean: mx.array | None = None,
    bn_var: mx.array | None = None,
    bn_eps: float = 1e-4,
) -> mx.array:
    """Unpack mflux's packed FLUX.2 latent into NHWC (B, H, W, 32) for TAEF2.

    Mirrors the unpack in mflux's flux2_klein.py:117-118 and the internals of
    Flux2VAE.decode_packed_latents / _unpatchify_latents, but outputs NHWC
    instead of feeding into the full VAE decoder.

    Args:
        packed: in-loop latent from mflux with shape (B, latent_height*latent_width, 128).
            This is the value of `latents` after each scheduler step in the denoise
            loop (i.e. what is passed to InLoopCallback.call_in_loop).
        latent_height: spatial height of the DiT latent grid = image_height // 16.
        latent_width: spatial width of the DiT latent grid = image_width // 16.
        bn_mean: optional BatchNorm running mean from Flux2VAE.bn.running_mean,
            shape (128,).  Pass None to skip de-normalization (identity BN).
        bn_var: optional BatchNorm running variance from Flux2VAE.bn.running_var,
            shape (128,).  Pass None to skip de-normalization (identity BN).
        bn_eps: epsilon used in Flux2VAE BatchNorm (default 1e-4, matching mflux source).

    Returns:
        NHWC array of shape (B, latent_height*2, latent_width*2, 32) in raw
        TAEF2 latent space, ready for TAEF2.decode().
        - latent_height*2 = image_height // 8
        - latent_width*2  = image_width  // 8

    Shape example (1024×1024 image):
        packed: (1, 4096, 128)  →  returns (1, 128, 128, 32)

    Notes:
        If bn_mean/bn_var are not provided, the function assumes identity
        BatchNorm (mean=0, var=1). This gives the correct *shape* but may have
        slight value offset compared to real mflux latents. For a preview
        thumbnail this is acceptable; for exact fidelity pass the BN stats from
        Flux2VAE.bn.
    """
    if packed.ndim != 3:
        raise ValueError(f"Expected packed latent with ndim=3, got shape={packed.shape}")

    batch_size, seq_len, channels = packed.shape
    expected_seq = latent_height * latent_width
    if seq_len != expected_seq:
        raise ValueError(
            f"packed seq_len mismatch: got {seq_len}, expected "
            f"{expected_seq} (latent_height={latent_height}, latent_width={latent_width})"
        )
    if channels != 128:
        raise ValueError(f"Expected 128 channels in packed latent, got {channels}")

    # Step 1: reshape + transpose to NCHW 128-channel
    # (B, lH*lW, 128) -> (B, lH, lW, 128) -> (B, 128, lH, lW)
    x = packed.reshape(batch_size, latent_height, latent_width, channels)
    x = x.transpose(0, 3, 1, 2)  # (B, 128, lH, lW)

    # Step 2: BN de-normalize (mirrors Flux2VAE.decode_packed_latents lines 46-48)
    # raw = packed * bn_std + bn_mean
    if bn_mean is not None and bn_var is not None:
        bn_mean_ = mx.reshape(bn_mean, (1, -1, 1, 1))  # (1, 128, 1, 1)
        bn_std_ = mx.sqrt(mx.reshape(bn_var, (1, -1, 1, 1)) + bn_eps)
        x = x * bn_std_ + bn_mean_
    # else: identity BN assumed (mean=0, var=1 → std=1), x unchanged

    # Step 3: unpatchify (mirrors Flux2VAE._unpatchify_latents lines 53-58)
    # (B, 128, lH, lW) -> (B, 32, lH*2, lW*2)
    x = mx.reshape(x, (batch_size, channels // 4, 2, 2, latent_height, latent_width))
    # -> (B, 32, 2, 2, lH, lW)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    # -> (B, 32, lH, 2, lW, 2)
    x = mx.reshape(x, (batch_size, channels // 4, latent_height * 2, latent_width * 2))
    # -> (B, 32, lH*2, lW*2)  NCHW

    # Step 4: NCHW -> NHWC for TAEF2
    x = mx.transpose(x, (0, 2, 3, 1))  # (B, lH*2, lW*2, 32)

    return x
