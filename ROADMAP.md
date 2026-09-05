# mlx-taef Roadmap

A non-binding sketch of where the library is headed. Each item lists status, effort, and key risks so we can re-rank when priorities change.

## Released

- **v0.8.1** (2026-09-05) — mflux 0.19.x compatibility. mflux 0.19.0 added a `control_images` argument to its before-loop callback hook and passes it to every registered callback, so a generation with a `LivePreviewCallback` registered failed with a `TypeError` before its first denoise step; the hook now accepts it, and absorbs whichever conditioning keyword mflux adds next. The `mflux` extra widens to `>=0.17,<0.20`. Verified against mflux 0.19.1 (MLX 0.32.2): the in-loop callback contract, every packed-latent layout, and the committed parity fixtures are unchanged, and a live preview composes with mflux's new `--pid-decode` (NVIDIA PiD) final decode.
- **v0.8.0** (2026-08-09) — Krea 2 Turbo live preview. A new `Krea2` model decodes Krea 2 Turbo latents by reusing the taew2.1 weights already converted for `QwenImage` (Krea 2 generates on the Qwen-Image stack and shares its Wan 2.1 VAE), so there is no new download. Decode quality against mflux's full Krea 2 VAE is gated by an SSIM check (measured 0.9678). The showcase report and COMPARISON.md now report LPIPS alongside SSIM for every decode scenario. `ZImage.encode()` gets its own SSIM-gated validation against the full Z-Image VAE encoder (measured 0.9580). `mlx-teacache` now installs on Python 3.10 as well as 3.11+, so the showcase's combined scenario runs on every supported Python version. The live-preview integration is verified against mflux 0.18.1 with no code changes needed, and every model-loading showcase and benchmark subprocess now runs under the active-memory watchdog.
- **v0.7.1** (2026-07-25) — Python 3.10 support. The floor drops from 3.11 to 3.10 and the CI matrix covers 3.10 through 3.14, so no advertised version runs untested. `mlx-teacache` (a showcase/test dependency that needs 3.11) is gated by a version marker; the runtime dependencies all work on 3.10.
- **v0.7.0** (2026-07-24) — preview resilience and hardening. `LivePreviewCallback` now isolates runtime preview failures by default and offers `on_error="raise"` for strict callers. All built-in weight files have immutable revisions and role-specific sha256 pins, including the public conversion command. Cache keys include a converter-format version, and CI uses the frozen lockfile. The six-scenario release check isolates live workers, requires complete repetition samples and preview galleries, exits nonzero on partial failure, and compares the migrated v0.6.2 report with source and distribution provenance recorded separately. The release report completed all scenarios with no regression.
- **v0.6.2** (2026-07-09) — hardening and accuracy release. `LivePreviewCallback` rejects `every < 1` and a half-set BN pair and resets its state between generations; the converted-weights cache invalidates when a source's pinned revision or sha256 changes (Qwen-Image re-converts once), and every conversion path enforces those pins. The decode benchmark now measures the decode step at steady state, in isolation from one-time model construction — the tiny decoders run ~30 ms per step (earlier releases reported ~180–260 ms because they timed model construction inside the decode window), a ~8–10× speedup over the full VAE decode — and COMPARISON / EXAMPLES are re-measured with two Z-Image scenarios added. Also fixes the mflux quickstart in the README.
- **v0.6.1** (2026-06-30) — maintenance release. Converted-weights cache writes are now atomic (temp file + rename), so an interrupted download or convert can no longer leave a truncated file for a later run to trust as valid. Python 3.14 is now supported, factory constructors return their concrete type, and the decoder/encoder role and preview variant are typed literals. Adds a linked live-preview demo page.
- **v0.6.0** (2026-06-23) — Qwen-Image / Qwen-Image-Edit live preview. A new `QwenImage` model ports madebyollin's taew2.1 tiny autoencoder (Wan 2.1 VAE, 16-channel latent) to pure MLX — the first variant on the new `taehv` architecture (2D convs with recurrent temporal blocks, run for a single still image). Adds `LivePreviewCallback(variant="qwen-image")` and `mlx-taef bench --variant qwen-image`. Decode and encode match the upstream reference to ~3e-6 (committed parity fixtures). Live-preview quality against the full Wan VAE is community-measured (the ~20B Qwen-Image model won't fit a usable resolution on 32 GB).
- **v0.5.1** (2026-06-20) — mflux 0.18.x compatibility. The `mflux` extra widens to install against mflux 0.18.x alongside 0.17.x (the previous `<0.18` pin excluded it); the live-preview integration is verified against mflux 0.18.0 with no API or behavior change.
- **v0.5.0** (2026-06-18) — live-preview ergonomics. `LivePreviewCallback` auto-detects `latent_height` / `latent_width` from the mflux generation config on each call, so the resolution no longer has to be passed by hand; passing both still overrides, and passing exactly one now raises. The auto-extracted Flux2VAE batch-norm `eps` is forwarded into the TAEF2 unpack so a non-default `eps` previews faithfully, and `auto_bn=True` on a non-TAEF2 variant now logs its no-op. Drops three showcase-only exceptions (`SchemaVersionError`, `FixtureLatentMissingError`, `MlxTeacacheNotInstalledError`) from the package-root exports; they remain importable from `mlx_taef.errors`.
- **v0.4.2** (2026-06-14) — hardening patch. `decode()` / `encode()` reject non-4-D inputs with a `ValueError` naming the expected NHWC rank, closing a gap in the v0.3.1 channel-count guard. Internal: the release workflow's `download-artifact` action moves to a Node 24 release; the `mflux_live_preview` example now prints per-step decode timing next to the full-VAE final decode (thanks @ianscrivener, #20).
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

- **`decode_streaming()` API.** Incremental decode for very-low-memory environments. Speculative.
- **JPEG XL output** from `decode_image()` for compressed previews. Browser support is the bottleneck; revisit when Safari/Chrome both support animated JXL.

## Known-upstream variants, not yet supported

These exist in upstream [`madebyollin/taesd`](https://github.com/madebyollin/taesd) but are not in mlx-taef as of v0.2.3. Each requires its own integration + calibration cycle.

- **TAESD3** — for Stable Diffusion 3 / 3.5.
- **TAESANA** — for Sana.
- **TAESDV** — for SD video models.
- **TAEHV** — for Hunyuan / Wan / CogVideoX. The Wan 2.1 decoder (taew2.1) ships as of v0.6.0 for Qwen-Image; the multi-frame video models on this architecture are not yet wired.
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
