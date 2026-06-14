# mlx-taef Roadmap

A non-binding sketch of where the library is headed. Each item lists status, effort, and key risks so we can re-rank when priorities change.

## Released

- **v0.4.1** (2026-06-14) — documentation and packaging accuracy pass, no code changes. The Z-Image SSIM calibration is now described correctly (it runs as an opt-in network test, not in default CI); the README's no-PyTorch note is scoped to the base install (the `mflux` extra follows mflux's dependencies, which include PyTorch); and the source distribution is an allowlist that no longer ships the fixtureless test suite or local tool state. The wheel is unchanged.
- **v0.4.0** (2026-06-13) — Z-Image / Z-Image-Turbo support. A new `ZImage` variant reuses TAEF1's FLUX.1 autoencoder weights (Z-Image shares FLUX.1's 16-channel latent contract), so previewing Z-Image needs no new download; validated by an SSIM ≥ 0.75 calibration against mflux's full Z-Image VAE (measured 0.94; the check runs as an opt-in network test, not in default CI). Adds `LivePreviewCallback(variant="zimage")`, `mlx-taef bench --variant zimage`, a Z-Image showcase scenario, and a top-level EXAMPLES.md. Decode / live preview is the validated path; the inherited `encode()` reuses the TAEF1 encoder on the shared contract and is best-effort.
- **v0.3.1** (2026-06-08) — hardening patch. `decode()`/`encode()` raise `TaefError` when called before weights are loaded, and raise `ValueError` naming the variant when a latent has the wrong channel count or an image isn't RGB. Importing `mlx_taef.integrations.mflux` without mflux installed raises the new `MfluxNotInstalledError` (`TaefError` + `ImportError`). Test fixes: a conv-transpose test that now fails if the transpose is removed, offline CLI role-routing and collection-gating coverage, and a wired-limit cap test that no longer mutates process state.
- **v0.3.0** (2026-06-06) — internal kernel refactor, no public API change. Each variant is now a composable `ModelKernel` (`mlx_taef.kernels`), so adding a model is a self-contained entry instead of edits spread across `variants.py` / `api.py` / `convert.py`; `variants.py` stays a back-compat shim. Ships one user-facing fix: the mflux `LivePreviewCallback` FLUX.1 path fed mflux's packed latent straight to the decoder and produced wrong previews, and now unpacks correctly. The converted-weights cache is re-keyed on the weight source, so 0.2.x caches rebuild once on first run.
- **v0.2.4** (2026-06-03) — internal hardening, no user-facing change: the showcase regression gate now guards peak-memory and the TeaCache `skipped_count` (and a dropped metric/block), and the error/bench tests now exercise real raise conditions instead of just class declarations. Test suite and `scripts/` only; the published wheel is unchanged.
- **v0.2.3** (2026-05-29) — strict weight loading: `from_pretrained_local` raises on an incomplete or wrong-shaped weights file instead of loading a silently-wrong model (new `ConversionError`); the HF→MLX converter validates parameter coverage and shapes at convert time; end-to-end parity tests gate on an absolute pixel/latent tolerance instead of cosine similarity; a bare `pytest` skips network and benchmark tests by default, with `--run-network` / `--run-benchmark` opt-ins.
- **v0.2.0** (2026-05-27) — `LivePreviewCallback` auto-bn extraction for taef2; [COMPARISON.md](COMPARISON.md) + `scripts/run_showcase.py` measured showcase (4 scenarios); subprocess-per-rep `scripts/bench_decode.py`; per-variant `memory_cap_hint_gb` + `FULL_VAE_CAP_GB`; session-level `tests/conftest.py` MLX cap; cross-process MLX non-determinism caveat in [docs/manual-verification.md](docs/manual-verification.md); Trove classifier alignment; `showcase` runtime extra.
- **v0.1.1** (2026-05-13) — sha256 sidecars on committed fixtures; SSIM ≥ 0.75 perceptual contract on roundtrip; release workflow ships a GitHub Release on tag push.
- **v0.1.0** (2026-05-13) — initial public release. 4 variants (TAESD, TAESDXL, TAEF1, TAEF2) with one consistent API. mflux `LivePreviewCallback`. 99-100% coverage.

## Active

(Empty.)

## Future improvements (no fixed release target)

- **LPIPS as a second perceptual metric alongside SSIM.** SSIM is structural; LPIPS catches the perceptual hallucination TAESD-family models produce that SSIM under-weights. Costs `pip install lpips` + 2 lines in the orchestrator. Ship when ROADMAP item count justifies.
- **`decode_streaming()` API.** Incremental decode for very-low-memory environments. Speculative.
- **JPEG XL output** from `decode_image()` for compressed previews. Browser support is the bottleneck; revisit when Safari/Chrome both support animated JXL.
- **Auto-detect resolution from mflux flux instance.** `LivePreviewCallback(flux=...)` could auto-derive `latent_height` / `latent_width` from `flux.config` instead of requiring the user to pass them.

## Known-upstream variants, not yet supported

These exist in upstream [`madebyollin/taesd`](https://github.com/madebyollin/taesd) but are not in mlx-taef as of v0.2.3. Each requires its own integration + calibration cycle.

- **TAESD3** — for Stable Diffusion 3 / 3.5.
- **TAESANA** — for Sana (the `f32` arch_variant placeholder in `variants.py` is reserved for this).
- **TAESDV** — for SD video models.
- **TAEHV** — for Hunyuan / Wan / CogVideoX.
- **TAEM1** — for Mochi 1.

The mlx-taef integration cost for each is roughly: half-day of code (new variant entry, weight conversion path, smoke test) + a calibration/reference-fixture pass to verify the SSIM/parity bound holds. Not blocked by anything; awaits user/community demand.

## Out of scope (deliberate non-goals)

- **PyTorch backend.** mlx-taef is MLX-only. The upstream `madebyollin/taesd` already covers PyTorch.
- **Live-preview UI rendering.** mlx-taef writes a webp file to disk; downstream apps (mflux CLI, custom notebooks) render it.
- **Server / API layer.** This is a library, not a service.

## How to use this doc

1. **Active items first.** Items under `## Active` are committed. Finish current Active item before pulling the next in.
2. **Future improvements next.** Items under `## Future improvements` are pre-vetted improvement ideas. Each can be lifted into an Active release.
3. **Known-upstream variants are a menu, not a queue.** Pick from it based on community demand + bench cost.
4. **Out of scope is durable.** Re-opening an item there requires evidence that the original reasoning no longer holds.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
