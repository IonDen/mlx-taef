# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-08-09

Krea 2 Turbo live preview.

### Added
- `Krea2` decodes Krea 2 Turbo latents. Krea 2 generates on the Qwen-Image stack and shares its
  Wan 2.1 VAE, so this variant reuses the taew2.1 weights already converted for `QwenImage` — one
  shared converted-cache entry, no new download. Construct it with `Krea2.from_pretrained()`, or
  preview a live generation with `LivePreviewCallback(variant="krea2")`. Adds
  `mlx-taef bench --variant krea2`. Decode quality against mflux's full Krea 2 VAE is gated by an
  opt-in SSIM check on a committed fixture (measured 0.9678, floor 0.75).
- The showcase report and COMPARISON.md now carry LPIPS alongside SSIM for every decode scenario.
  LPIPS is a learned perceptual distance (lower is better) and catches artifacts SSIM's structural
  comparison under-weights.

### Changed
- `mlx-teacache`, used by the `showcase` extra and the `test` dependency group, now installs on
  Python 3.10 as well as 3.11+, since mlx-teacache 0.9.3 dropped its own 3.11 floor. The combined
  showcase scenario and its tests now run on every Python version this project supports.
- `ZImage.encode()` is now validated the same way decode already was: an opt-in cross-roundtrip
  test (TAEF1 encode into the full Z-Image VAE's decoder) measures SSIM 0.9580 against the same
  0.75 floor.
- The live-preview integration is verified against mflux 0.18.1: the callback contract, the
  packed-latent layouts, and the batch-norm stats the auto-bn path reads are unchanged, so
  `mlx-taef[mflux]` needs no code changes. The `mflux` extra pin stays `>=0.17,<0.19`.
- Every model-loading showcase and benchmark subprocess, not only the three live-generation
  workers, now runs under the active-memory watchdog. A decode-only rep that starts paging aborts
  with an honest artifact instead of risking the machine.
- CI's dependency groups are now installed in isolation (`[tool.uv] default-groups = []`): each
  job's `uv sync --frozen --group <name>` installs exactly that group instead of also pulling in
  uv's implicit default group. This uncovered a missing Pillow dependency in the `typecheck`
  group — `mypy --strict` needs PIL's types to check `integrations/mflux.py` — now declared there
  directly.

### Internal
- The README links a live-preview GIF near the top: TAEF1 previews animating step by step next to
  the finished full-VAE decode held static, generated with the new `scripts/make_preview_gif.py`.

## [0.7.1] - 2026-07-25

### Changed

- Python 3.10 is now supported (`requires-python` drops from `>=3.11` to `>=3.10`). The CI test
  matrix runs 3.10 through 3.14, so every advertised version is exercised by the full offline
  suite, including the bit-exact parity fixtures. On 3.10 the `Self` return types come from
  `typing-extensions`, which installs automatically on that version only.
- The `showcase` extra and the `test` group install `mlx-teacache` only on Python 3.11+, since it
  requires 3.11. Everything else in both, and all of the runtime dependencies, work on 3.10; the
  showcase's combined scenario already reports a clear error when `mlx-teacache` is absent.

## [0.7.0] - 2026-07-24

This release makes live previews resilient to runtime errors and closes the remaining hardening work across downloads, conversion, errors, and the benchmark harness.

### Added
- `LivePreviewCallback(..., on_error="disable" | "raise")` controls runtime preview failures. The default logs one warning and disables previews for the rest of the current generation, so a preview problem does not discard a long-running image. Strict integrations can use `on_error="raise"` to keep the previous fail-fast behavior.
- Every built-in Hugging Face weight file now has an immutable revision and role-specific sha256 pin. The opt-in network test downloads, verifies, converts, and loads every decoder and encoder source through the runtime path.
- `UnknownArchitectureError` gives architecture lookup failures the same clean, package-rooted error surface as unknown variants.

### Fixed
- TAEF2 auto-BN now respects the VAE's epsilon, rejects incomplete explicit BN pairs, and warns when `auto_bn` cannot resolve stats. FLUX.1, FLUX.2, and Qwen packed-latent unpacking now validates the sequence length before reshape.
- A live-preview decode or file-write failure no longer terminates a generation under the default callback policy. Callback state resets on the next generation.
- Four-dimensional convolution weights are always transposed during conversion. Ambiguous equal-sized channel dimensions can no longer skip the required layout change.
- `UnknownKernelError` and `UnknownArchitectureError` render without `KeyError`'s extra quote layer. Direct `Taef()` construction, memory-cap lookup, and `mlx-taef info` failures now return clear, stable errors.
- The documented direct invocation of `scripts/diff_showcase_report.py` works from any current directory.
- `mlx-taef convert` now uses the kernel registry and the runtime download cache, so its downloads enforce the same immutable revision and sha256 checks as `from_pretrained()`.
- The showcase command exits nonzero if any scenario fails after checkpointing the remaining results. Decode medians now require every requested repetition, and live scenarios require one non-empty preview per inference step plus a non-empty final image.

### Changed
- Converted-weight cache keys include a converter-format version as well as the source revision and digest. The first load after this upgrade rebuilds each converted cache once. Converted cache directories are created with owner-only permissions.
- CI installs from the lockfile with `uv sync --frozen` on every supported Python version.
- Live showcase generations run in separate worker processes. Each worker has a 55-minute wall budget and a 28 GiB active-memory ceiling on the 32 GiB reference Mac, writes an abort record before exit, and leaves completed scenario results checkpointed. The report schema is now version 2 and records generation and tiny-decoder dtypes separately.
- Benchmark metadata records both the source-derived git version and the installed distribution version. The report loader migrates the committed v0.6.2 schema when comparing releases, and the regression checker also guards the number of preview frames.
- The benchmark report was re-measured on Apple M1 Max at commit `1e79c29`. All six scenarios completed with every requested repetition and preview frame, and the regression checker found no latency, memory, SSIM, TeaCache, or gallery regression against v0.6.2.

### Internal
- The kernel registry is the source of truth for mid-block GroupNorm and memory-cap metadata. The old `MIDBLOCK_GN` mapping remains as a derived compatibility view.
- Small integration seams now have behavioral coverage for callback registration docs, numbered frames, missing bindings, error formatting, kernel metadata, CI lockfile use, and benchmark schema handling.

## [0.6.2] — 2026-07-09

A hardening and accuracy release. It hardens the live-preview callback and the converted-weight cache, and corrects the published decode-benchmark numbers after fixing how they were measured.

### Fixed
- The README's mflux live-preview quickstart called a `Flux2Klein` constructor that does not exist in any supported mflux version; it now uses the working `Flux2Klein(quantize=4, model_config=...)` form.
- `LivePreviewCallback` rejects `every < 1` and a half-set BN pair (only `bn_mean` or only `bn_var`) at construction, rather than dividing by zero mid-generation or discarding the BN stats without a word.
- `LivePreviewCallback` resets its step counter and frame gallery at the start of each generation. Reusing one callback across several `generate_image` calls no longer misaligns the `every` cadence or mixes preview frames from different runs.
- The converted-weights cache key now includes a source's pinned revision and sha256, so bumping a pin re-converts instead of serving stale weights. Qwen-Image is the only pinned model today, so its users re-convert once on upgrade; the other models keep their existing cache.
- Every conversion path now enforces a source's `revision` pin and, for single-file sources, verifies the `sha256`. Previously only the taew2.1 path did.

### Changed
- The decode benchmark now times the decode step at steady state, with model construction moved outside the timed window and one untimed warmup call before the clock starts. The tiny decoders run about 30 ms per step — the earlier releases' ~180–260 ms figure timed one-time model construction inside the decode window — for a **~8–10×** speedup over the full VAE decode. COMPARISON and EXAMPLES are re-measured, and the Z-Image decode and live-preview scenarios join the showcase.
- The showcase harness writes its report after each scenario and records a failing scenario as an error instead of aborting the run. The report differ now flags a scenario or metric that vanishes between runs.

### Internal
- The `from_pretrained` `repo_id` mismatch guard is now tested; the latent-capture path is covered against the real mflux callback registry; shipped docstrings state the memory-cap constraint directly instead of pointing at a repo-local file; the local coverage gate matches CI at 95%.

## [0.6.1] — 2026-06-30

A maintenance release: a cache-corruption fix, Python 3.14 support, sharper public-API types, and a live-preview demo.

### Added
- Python 3.14 is now tested in CI and advertised in the package classifiers.
- A short live-preview clip — TAEF1 decoding a FLUX.1-dev generation step by step — linked near the top of the README.

### Fixed
- Converted weights are written atomically (temp file + rename), so a run interrupted mid-download/convert can no longer leave a truncated file in the cache. (If an earlier version already left a corrupt cache file, clear it once with `rm -rf ~/.cache/mlx-taef/` — existing files are not auto-repaired.)

### Changed
- Factory constructors (`from_pretrained`, `from_pretrained_local`, `from_kernel`) are now typed to return the concrete subclass, and the decoder/encoder role and the preview variant are typed literals. Editor autocomplete and type-checking improve; runtime behavior is unchanged.
- Internal: the test suite was hardened (parity-oracle integrity pinning, offline-by-default mflux tests).

## [0.6.0] — 2026-06-23

Qwen-Image live preview.

### Added
- `QwenImage` decodes Qwen-Image and Qwen-Image-Edit latents — the Wan 2.1 VAE's 16-channel
  latent — with a pure-MLX port of madebyollin's taew2.1 tiny autoencoder. Construct it with
  `QwenImage.from_pretrained()` for standalone decode/encode, or preview a live generation with
  `LivePreviewCallback(variant="qwen-image")`. Adds `mlx-taef bench --variant qwen-image`.

### Notes
- Decode and encode match the upstream taew2.1 reference to within ~3e-6 (fp32, measured worst),
  gated by committed parity fixtures for both paths.
- taew2.1 is a different shape from the rest of the family: a 2D-conv autoencoder with recurrent
  temporal blocks, run here for a single still image. It is the first variant on the new `taehv`
  architecture.
- The weights are a sha256-verified re-host of madebyollin's canonical taew2.1, which is published
  on GitHub only; the kernel pins the file by its hash.
- Live-preview quality against the full Wan VAE is community-measured: Qwen-Image is a ~20B model
  that does not fit a usable resolution on 32 GB, so that comparison is not captured here.

## [0.5.1] — 2026-06-20

A compatibility release. No API or behavior changes.

### Changed
- The `mflux` extra now installs against mflux 0.18.x as well as 0.17.x. mflux 0.18.0
  shipped after the previous pin, so the old `<0.18` bound left anyone already on 0.18
  unable to install `mlx-taef[mflux]` without downgrading mflux. The pin is now
  `>=0.17,<0.19`. The live-preview integration was verified against mflux 0.18.0: the
  callback contract, the generation config the auto-resolution reads, the packed-latent
  layout, and the Flux2VAE batch-norm stats the auto-bn path extracts are unchanged, so
  no code needed to change.
- The `showcase` extra moves its `mlx-teacache` pin to `>=0.9.1,<0.10`, the first
  mlx-teacache release that supports mflux 0.18.x, so `mlx-taef[showcase]` stays coherent
  on mflux 0.18.

## [0.5.0] — 2026-06-18

A live-preview ergonomics release.

### Added
- `LivePreviewCallback` auto-detects the preview resolution. Leave `latent_height` /
  `latent_width` unset and the callback reads the image size from the mflux generation config at
  run time, so a non-square or non-512 render previews correctly without you passing dimensions
  by hand. Passing both dimensions still overrides the auto-detection.

### Changed
- `latent_height` / `latent_width` now default to `None` (auto-detect) instead of `32`. Callers
  that pass both explicit dimensions are unaffected; callers that relied on the old `32` default
  now get the right dimensions for their actual resolution. Passing exactly one of the two now
  raises a `ValueError` (set both, or leave both to auto-detect); previously the unset one
  silently fell back to `32`.
- The auto-extracted Flux2VAE batch-norm `eps` is forwarded into the TAEF2 preview unpack, so a
  VAE whose `eps` differs from the `1e-4` default previews faithfully.
- `auto_bn=True` on a non-TAEF2 variant now logs that it is a no-op (it was silent before), and
  the `auto_bn` / `latent_height` / `latent_width` arguments are documented on the callback.

### Removed
- `SchemaVersionError`, `FixtureLatentMissingError` and `MlxTeacacheNotInstalledError` are no
  longer exported from the package root. They are raised only by the bundled showcase script and
  remain importable from `mlx_taef.errors`.

## [0.4.2] — 2026-06-14

A small hardening patch.

### Changed
- `decode()` and `encode()` now reject inputs that aren't 4-D NHWC arrays, raising a `ValueError`
  that names the expected rank instead of failing deep in the conv stack with an opaque error.
  This closes a gap in the v0.3.1 channel-count guard, which a wrong-rank input could slip past
  (for example a 3-D array whose last dimension happened to match the channel count).
- The `mflux_live_preview` example now prints per-step TAEF2 decode timing next to the full
  Flux2VAE final decode, so the preview-vs-full-VAE speed difference is visible when you run it.
  Thanks to @ianscrivener (#20).

### Internal
- The release workflow's `actions/download-artifact` step moves to a Node 24 release (v8),
  clearing the Node 20 deprecation warning.

## [0.4.1] — 2026-06-14

A documentation and packaging accuracy pass. No code or model behavior changes.

### Changed
- The Z-Image SSIM ≥ 0.75 calibration is now described correctly across the docs. It runs as an
  opt-in network test (`pytest --run-network`), not in default CI, so the docs no longer call it
  CI-gated.
- The README's "no PyTorch" note is scoped to the base install. The optional `mflux` extra
  follows mflux's dependency set, which currently includes PyTorch, so `mlx-taef[mflux]` brings
  it in.

### Fixed
- The source distribution is now an allowlist of the package, user-facing docs, examples, and
  license. It no longer ships the test suite without the parity fixtures those tests need, and a
  local `uv build` can no longer sweep machine-local tool state or generated artifacts into the
  archive. The wheel is unchanged (package-only).

## [0.4.0] — 2026-06-13

Z-Image / Z-Image-Turbo support. Z-Image's VAE shares FLUX.1's 16-channel latent contract, so
the existing TAEF1 decoder previews it with no new weights to download. Validated by an SSIM ≥
0.75 calibration against mflux's full Z-Image VAE (measured 0.94). That calibration runs as an
opt-in network test (`pytest --run-network`), not in default CI.

### Added
- `ZImage` — a model class for Z-Image / Z-Image-Turbo live preview. Reuses TAEF1's weights and
  its converted-weights cache (no separate download). Loads via the standard API
  (`from_pretrained` / `from_pretrained_local` / `from_kernel`).
- `LivePreviewCallback(variant="zimage", ...)` for previewing mflux Z-Image generations.
- `mlx-taef bench --variant zimage`.
- Top-level [EXAMPLES.md](EXAMPLES.md): a narrative walkthrough of live preview and low-memory
  decode for each model, with captured frames and the measured cost of each decode.

### Notes
- The validated path is decode / live preview (the SSIM calibration runs as an opt-in network
  test, not in default CI). `ZImage`
  inherits `encode()`, which reuses the TAEF1 encoder on the shared latent contract; it is not
  separately validated against Z-Image's distinct VAE encoder, so encode / img2img is best-effort.
- Measured on Apple M1 Max (32 GB), mflux 0.17.5 / MLX 0.31.2, int4: TAEF1 decodes a Z-Image
  latent in ~62 ms versus ~2.76 s for the full Z-Image VAE, at ~0.55 GB versus ~1.96 GB peak.
  Reproduce with `uv run python scripts/run_showcase.py --scenario zimage_vs_vae`.

## [0.3.1] — 2026-06-08

Hardening patch. No public API shape changes; the one new symbol is `MfluxNotInstalledError`.
The existing parity and SSIM fixtures gate the change bit-for-bit.

### Added
- `MfluxNotInstalledError` — raised when `mlx_taef.integrations.mflux` is imported
  without mflux installed. Subclasses both `TaefError` and `ImportError`, so
  `except ImportError` keeps working and `except TaefError` now catches it too.

### Changed
- `decode()` / `decode_image()` raise `TaefError` when called before decoder weights
  are loaded (for example on a directly constructed `TAEF2()`), and `encode()` raises
  when the model was loaded without an encoder — instead of running a random-init module
  and returning garbage. Normal `from_pretrained` / `from_pretrained_local` usage is
  unaffected.
- A latent with the wrong channel count in `decode()`, or a non-RGB image in `encode()`,
  now raises `ValueError` naming the variant and the expected vs actual channel count,
  rather than an opaque MLX conv shape error.

### Fixed
- The error-class docstring no longer references an exception symbol that never existed.
  Every exception exported from the package root is now either raised by importable code
  or documented as raised by the bundled showcase tooling.

## [0.3.0] — 2026-06-06

Internal refactor. No change to the public API or to decoded output: `TAESD`,
`TAESDXL`, `TAEF1`, `TAEF2`, `from_pretrained` / `from_pretrained_local`,
`get_memory_cap_hint`, and imports from `mlx_taef.variants` all behave as before. The
existing parity and SSIM fixtures gate the change bit-for-bit.

### Changed
- Each variant is now a composable model kernel (`mlx_taef.kernels`): a frozen
  `ModelKernel` built from an `ArchSpec`, a `ConversionStrategy` (which owns the whole
  HF→MLX conversion), a `LatentSpec`, a `WeightSource`, and an optional `MfluxBinding`.
  Adding a model is a self-contained kernel entry instead of edits scattered across
  `variants.py`, `api.py`, and `convert.py`; `variants.py` stays as a back-compat shim.
- The converted-weights cache is keyed on the weight source (repo, filename, role)
  rather than the variant name, so two models that share upstream weights share one cache
  entry. Caches written by 0.2.x are ignored and rebuilt once on the first run after upgrade.

### Fixed
- FLUX.1 live previews through the mflux `LivePreviewCallback` were wrong. During the
  denoise loop mflux hands the callback a packed `(B, N, 64)` latent, but the old code
  expected an unpacked `(B, 16, H, W)` and let the packed latent fall through to the
  decoder. The callback now unpacks per model via the kernel binding, matching mflux's
  own `unpack_latents`, and a test pins the result against it.

## [0.2.4] — 2026-06-03

Internal hardening only. No change to the library API, runtime behavior, or the
published wheel; every change is in the test suite and developer tooling
(`scripts/`), neither of which ships in the package.

### Changed
- The showcase regression gate (`scripts/diff_showcase_report.py`) now also guards
  the headline peak-memory metric and the TeaCache `skipped_count`, and flags a
  baseline metric or block disappearing from a new report rather than only a worse
  number (new `--memory-tolerance`, default 0.10). Previously a memory regression or
  a dropped skip count passed the gate silently.

### Tests
- The error tests drive real bad inputs through each package error's raise site
  (`ConversionError`, `SchemaVersionError`, `FixtureLatentMissingError`) instead of
  only asserting the class hierarchy, and the bench harness's `_resolve_cap_gb`
  unknown-condition guard gained the raise test it was missing. Each new
  raise-condition test was mutation-verified.

## [0.2.3] — 2026-05-29

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
- A bare `pytest` (or `uv run pytest`) now skips the `network` and `benchmark`
  tests by default, so a local run is fast and offline and matches what CI runs.
  Opt back in with `--run-network` (real HF downloads) or `--run-benchmark` (perf
  timings). Previously the only deselection lived in the CI invocation, so a bare
  local run attempted the real downloads and the benchmark timings.

### Removed
- Unused `slow`, `gpu`, and `integration` pytest markers, which were registered
  but applied to zero tests. `network` and `benchmark` remain.

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
- `mlx_taef._memory_caps` — device-aware wired+memory cap helper. Computes `(wired_gb, memory_gb)` from `mx.device_info()["max_recommended_working_set_size"]` and clamps the 20 GB / 22 GB targets below the device ceiling. On a 32 GB M1 Max it returns `(20, 22)` unchanged; on smaller CI runners it returns a smaller pair so `set_wired_limit` won't raise.
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
