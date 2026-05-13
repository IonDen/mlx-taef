# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-13

### Added
- Public release of `mlx-taef`.
- Pure MLX implementation of the TAESD family architecture: `Clamp`, `Block` (with optional `midblock_gn` pool branch using `pytorch_compatible=True` GroupNorm), `make_decoder`, `make_encoder`.
- Four variants under one consistent API: `TAESD`, `TAESDXL`, `TAEF1`, `TAEF2`.
- `Taef.from_pretrained()` — auto-download from HF Hub, convert to MLX, cache at `~/.cache/mlx-taef/`. Zero PyTorch dependency at runtime.
- `Taef.from_pretrained_local()` — load from already-converted local safetensors.
- `Taef.decode()` / `decode_image()` / `encode()` / `scale_latents()` / `unscale_latents()`.
- Weight conversion handling both upstream-Sequential and Diffusers key formats (with the +1 decoder index offset for Diffusers).
- CLI: `mlx-taef convert --variant <name> --role {decoder,encoder} --dst <path>`, `info <path>`, `bench --variant <name>`.
- `mlx_taef.integrations.mflux.LivePreviewCallback` for live previews during FLUX.1 / FLUX.2 Klein generation in mflux.
- `mlx_taef.integrations.mflux.unpack_flux2_latent` helper handling mflux's packed DiT latent layout + optional BN denormalization.
- 69 tests, 100% line + branch coverage. CI on macOS 14 ARM64 (Python 3.11/3.12/3.13) plus Linux lint/typecheck.
- Tier 1 parity tests for every MLX layer vs PyTorch reference.
- Tier 2 parity: pre-baked reference fixtures (committed to repo); decoder cosine sim > 0.999 on 5 fixtures per variant, encoder cosine sim 1.00000000.
- Tier 3 property tests via hypothesis (shape invariants, dtype propagation, determinism, roundtrip).
- Tier 5 perf budget: decode 1024×1024 fp16 < 200 ms; peak memory < 1.5 GB.

### Known limitations
- `LivePreviewCallback` without `bn_mean`/`bn_var` produces structurally-correct but color-shifted previews. Pass the BN stats from `Flux2VAE.bn.running_{mean,var}` for exact recovery.
- bf16 dtype works but isn't hardware-accelerated on M1/M2 (only M3+).

## [0.0.2] — 2026-05-12

Phase 2: all four variants + encoder side + property tests.

## [0.0.1-alpha] — 2026-05-12

Phase 1: TAEF2 decoder MVP.
