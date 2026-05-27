# Audit: PR #1 mlx-taef v0.2.0 re-review

Source: `https://github.com/IonDen/mlx-taef/pull/1` (`feature/v0.2.0` at `fd654e2034b8e6c899aa277badb4f9ac470fdeb9`)
Date: 2026-05-27
Scope: Re-review of the updated PR after the prior audit findings were applied.

## Findings

### 1. The script help still shows a command that does not reproduce the committed benchmark policy
Severity: Medium
Refs: `scripts/run_showcase.py:14`, `scripts/run_showcase.py:129`, `COMPARISON.md:28`

Evidence: The `run_showcase.py` module docstring, which is used by argparse help, still says to run `uv run python scripts/run_showcase.py --scenario all --reps 3`. The actual benchmark policy is asymmetric: TAEF defaults to 5 reps and full VAE defaults to 3 reps. `COMPARISON.md` reports "median of 3/5 cold subprocess reps" and its reproducer correctly omits `--reps`.

Impact: Users following `scripts/run_showcase.py --help` will generate a report with only 3 TAEF reps, not the committed measurement protocol. That weakens reproducibility of the showcase numbers.

Fix: Remove `--reps 3` from the script docstring's canonical usage, or explicitly label it as an override that will not reproduce the committed 3/5 protocol.

### 2. `COMPARISON.md` still documents the removed private gallery callback
Severity: Medium
Refs: `COMPARISON.md:58`, `src/mlx_taef/integrations/mflux.py:151`, `scripts/run_showcase.py:242`, `CHANGELOG.md:21`

Evidence: The updated implementation folded gallery behavior into the public `LivePreviewCallback(numbered_frames=True)` API, and `run_showcase.py` now constructs that public callback. `CHANGELOG.md` documents the new public gallery mode. But `COMPARISON.md` still says `_GalleryPreviewCallback` decodes and saves the live-preview frames.

Impact: The public showcase doc now points readers at a private implementation detail that no longer exists. That hides the new public API surface and makes the documented benchmark flow disagree with the shipped code.

Fix: Replace `_GalleryPreviewCallback` in `COMPARISON.md` with `LivePreviewCallback(numbered_frames=True)` and keep the prose tied to the public callback path used by `run_showcase.py`.
