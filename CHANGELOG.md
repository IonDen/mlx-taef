# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ConversionError` (exported from `mlx_taef`) and a convert-time coverage + shape
  check in the HF→MLX weight converter. A conversion that fails to produce an
  expected model parameter, or produces one whose shape disagrees with the model,
  now raises and names the offending keys instead of silently writing an
  incomplete or wrong weights file.

### Changed
- `from_pretrained_local` now loads each submodule with `strict=True`. A weights
  file that is missing a parameter, or carries a wrong-shaped one, raises at load
  time instead of leaving the parameter at random init (a silently-wrong image).
  Loading is done per submodule so decoder-only loading (no `encoder_path`) stays
  valid — the encoder simply remains at init, as before.
- End-to-end decode and encode parity tests now gate on an absolute pixel/latent
  tolerance (`np.testing.assert_allclose`) instead of cosine similarity > 0.999.
  On the `[0, 1]` images these decoders produce, cosine similarity is DC-dominated
  and brightness/scale-insensitive — a +0.05 brightness shift scored cosine 0.9996
  and passed the old gate. The new gates are `atol=1e-4` for decode (measured worst
  maxabs ~1.1e-5) and `atol=1e-3` for encode (measured worst ~2.4e-4 on taef1);
  both use `rtol=0` because the values pass through zero. A regression test now
  asserts the decode gate rejects a +0.05 shift.

## [0.2.2] — 2026-05-27

Discoverability sweep. No runtime behavior changed; this release is a docs + metadata patch.

### Added
- `LICENSE` file at the repo root (MIT). The pyproject already declared `license = "MIT"` but the file itself was never committed, so GitHub's license auto-detect returned null and the README badge URL 404'd. Fixed.
- `examples/` directory with three runnable scripts:
  - `decode_flux_latent.py` — load a committed showcase latent, decode with TAEF2, write `out.webp`. Non-mflux use case.
  - `mflux_live_preview.py` — Flux2Klein + LivePreviewCallback with `flux=model` auto-bn. The headline integration.
  - `mflux_combined_with_teacache.py` — same as live_preview, also wrapped with `apply_teacache`. Combined-use story.
- README "Which library do I need?" section directly under the intro. Three-paragraph decision tree that converts arrivals from need-based searches.

### Changed
- PyPI `description` aligned with the GitHub repo description: now names live previews + low-memory decode + FLUX & SD targets instead of the previous generic "TAESD family on Apple MLX" framing.
- PyPI `keywords` expanded 6 → 14 (added `apple-silicon`, `diffusion`, `mflux`, `taef`, `taef1`, `taef2`, `tiny-autoencoder`, `vae`, `latent-preview`) so need-based PyPI search returns this package.
- PyPI `project.urls` adds `Source`, `Comparison`, and `Roadmap` entries pointing at the committed docs on `main`.

### Fixed
- PyPI `Documentation` URL removed (was pointing at `ionden.github.io/mlx-taef`, which returns HTTP 404). Replaced with the new `Source` / `Comparison` / `Roadmap` URLs that all resolve.
- README badge `License: MIT` now resolves (was 404 because the LICENSE file was missing — see Added above).

## [0.2.1] — 2026-05-27

Docs-only patch. No runtime behavior changed.

### Fixed
- README Status row now says "v0.2.0 — released on PyPI" instead of "v0.2.0 (in progress)" — the v0.2.0 PR landed before the tag pushed, leaving the row stale.
- README + ROADMAP no longer link to internal working-state documents. The Status row and Released row now stand on their own one-line summary instead of pointing readers at maintainer-only spec files that wouldn't render for someone browsing only the published surface.
- ROADMAP v0.2.0 row marked as released (2026-05-27).
- `unpack_flux2_latent` docstring no longer references a maintainer working note. Users were seeing the broken pointer in IDE tooltips when hovering the function in VS Code.

## [0.2.0] — 2026-05-27

Substantial release. Auto-bn extraction makes FLUX.2 live previews color-correct by default; the showcase bench publishes measured timings + perceptual fidelity numbers anyone can reproduce.

Headline measured results on M1 Max 32 GB (full table + reproducer in `COMPARISON.md`):

- TAEF2 vs full FLUX.2 VAE on the same latent: 8.3× faster decode (0.258 s vs 2.143 s median), 4.4× lower peak decode memory (0.59 GB vs 2.62 GB), SSIM 0.616.
- TAEF1 vs full FLUX.1 VAE: 11.0× faster decode (0.183 s vs 2.009 s median), 5.5× lower peak decode memory, SSIM 0.939.
- `live_preview` (FLUX.2 4-step + TAEF2 previews at every step): 11.26 s, peak 10.66 GB.
- `combined` (live_preview + mlx-teacache): 8.69 s, peak 7.90 GB — 1.30× faster than live_preview with 26% less peak memory.

### Added
- `LivePreviewCallback(flux=..., auto_bn=True)` — opt-out via `auto_bn=False`; auto-extracts `flux.vae.bn.running_mean` + `running_var` when `variant="taef2"` (the BN epsilon stays at the helper default `bn_eps=1e-4`, which matches mflux's `Flux2BatchNormStats` default at v0.17.5). Falls back to identity BN with a warning if the flux instance doesn't expose `.vae.bn`. New `callback.resolved_bn` tri-state attribute (`"explicit" | "auto" | "none"`).
- `LivePreviewCallback(numbered_frames=True)` — opt-in gallery mode that writes one image per step (`<stem>_step{NN}<ext>`) instead of overwriting a single path. Used by the v0.2.0 showcase to capture per-step progression; `callback.saved_paths` lists every written file.
- `TaesdVariantConfig.memory_cap_hint_gb` field + `get_memory_cap_hint(variant)` helper. Per-variant defaults: `taesd`/`taesdxl` None, `taef1` 1 GB, `taef2` 2 GB. Re-exported in `mlx_taef.__all__`.
- New exception classes in `src/mlx_taef/errors.py`: `TaefError` (root), `SchemaVersionError`, `MlxTeacacheNotInstalledError`, `FixtureLatentMissingError`. All re-exported in `mlx_taef.__all__`.
- `mlx_taef._memory_caps` — device-aware wired+memory cap helper. Computes `(wired_gb, memory_gb)` from `mx.device_info()["max_recommended_working_set_size"]` and clamps the CLAUDE.md targets (20 GB / 22 GB) below the device ceiling. On a 32 GB M1 Max it returns `(20, 22)` unchanged; on smaller CI runners it returns a smaller pair so `set_wired_limit` won't raise.
- `tests/conftest.py` session-level memory caps installed via the new `_memory_caps` helper (hardware-aware, not fixed 20/22 GB).
- `scripts/_caps.py` with `FULL_VAE_CAP_GB` shared constant (per-flux-variant cap for full-VAE baseline workers).
- `scripts/_capture_latent.py` — one-shot fixture-latent capture with sha256 sidecar.
- `scripts/bench_decode.py` — subprocess-per-rep decoder bench worker + orchestrator. `::BENCH_RESULT::` sentinel contract pinned (line-start, one-per-worker, JSON one-liner). Per-condition cap split: TAEF workers use the variant `memory_cap_hint_gb`; full-VAE workers use `FULL_VAE_CAP_GB[flux_variant]`. Failed-rep handling records errors and continues; raises if all reps fail.
- `scripts/run_showcase.py` — 4-scenario showcase orchestrator. Scenarios: `taef2_vs_vae`, `taef1_vs_vae`, `live_preview`, `combined`. SSIM computed in-orchestrator (cross-process safety per gotcha #24). JSON schema v1 with `importlib.metadata.version()` for version fields, per-rep arrays for timings + peak memory, per-condition `applied_cap_gb`, `ssim_per_pair` + `ssim_median`.
- `scripts/diff_showcase_report.py` — machine-enforced regression checker. Default tolerances: 10% wall-clock drift, 0.05 SSIM drop. Exits non-zero on any flagged regression.
- `COMPARISON.md` — showcase doc with 4 scenarios, ecosystem positioning, reproducer commands.
- `ROADMAP.md` — Released / Active / Future / Known-upstream-deferred (TAESD3, TAESANA, TAESDV, TAEHV, TAEM1).
- `docs/manual-verification.md` cross-process MLX non-determinism section.

### Changed
- `pyproject.toml`: Trove classifier alignment with mlx-teacache; new `showcase` runtime extra (`mflux`, `mlx-teacache`, `Pillow`, `scikit-image`); test dep group adds `mlx-teacache` and `scikit-image`.
- README install-pin example bumped `0.1.0` → `0.2.0`. `## Benchmarks` section now links to COMPARISON.md instead of inlining the (same-process v0.1.x) numbers.

### Removed
- Nothing intentionally removed. v0.1.x API surface preserved.

## [0.1.1] — 2026-05-13

### Added
- `tests/fixtures.toml` SHA-256 pins for all eight committed converted-weight fixtures plus the source image, with a `test_fixtures_integrity` test that verifies the hashes haven't drifted.
- `test_decode_handles_extreme_input_without_nan` — guarantees `decode()` produces finite, clamped-to-`[0, 1]` output even when fed pathologically large latents.
- `test_encode_decode_roundtrip_ssim_on_structured_image` — perceptual-quality contract via SSIM ≥ 0.75 (with luminance-MSE fallback when `scikit-image` is absent).
- `docs/release-setup.md` — one-time PyPI Trusted Publishing setup guide for the repo owner.
- `mlx-taef` now ships a GitHub Release on tag push (auto-generated notes, prerelease detection from tag suffix).

### Changed
- Release workflow (`release.yml`): added `github-release` job after the publish step.
- Parity tests now include inline comments documenting the cosine-similarity / `atol` tolerance choices.

### Fixed
- Linux typecheck previously failed because `# type: ignore[import-untyped]` didn't cover mypy's `import-not-found` error when `mflux` is absent. Broadened the ignore to cover both codes; `mflux` is now also installed in the `test` dependency group so the `LivePreviewCallback` test runs in CI and the integration code stays at honest coverage.
- Bumped `actions/checkout` v4 → v6, `actions/setup-python` v5 → v6 to clear the Node.js 20 deprecation warning.

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
