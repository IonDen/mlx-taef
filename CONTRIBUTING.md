# Contributing to mlx-taef

Thanks for your interest in mlx-taef. Bug fixes, new TAESD-family variants, documentation,
and example improvements are all welcome. Below is how to set up the project, the checks CI
runs, and the conventions worth knowing before you open a pull request.

## Development setup

The project uses [uv](https://github.com/astral-sh/uv). One command sets up the full dev
environment, including the test, lint, type-check, and docs tooling:

```bash
uv sync --all-groups
```

## Before you push: the checks CI runs

CI runs the lint and type checks on Linux and the test matrix on macOS (Python 3.11–3.13).
You can run the same checks locally:

```bash
uv run pytest                  # the test suite — fast and offline by default
uv run ruff check .            # linting
uv run ruff format --check .   # formatting (enforced, not just linting)
uv run mypy src                # type checking
```

Both `ruff check` and `ruff format --check` have to pass. Formatting is enforced, not just
linting. If either complains, `uv run ruff check --fix . && uv run ruff format .` fixes most
of it automatically.

A bare `pytest` skips the slow paths. Opt into them when your change touches that area:

```bash
uv run pytest --run-network     # tests that download weights from Hugging Face
uv run pytest --run-benchmark   # perf-timing tests (slow and noisy)
```

## The parity oracle: don't regenerate the fixtures

mlx-taef's regression guarantee is a set of committed, bit-exact reference outputs under
`tests/reference/`, `tests/converted/`, and `tests/fixtures.toml`. If a parity test fails,
that means your change altered the numbers — investigate the diff, don't "refresh" the
fixtures to make the test pass. The fixture generator exists only for deliberate, reviewed
updates, never as a way to clear a red parity test.

## Adding a new model

Each model is a self-contained kernel under `src/mlx_taef/kernels/`: an architecture spec, a
weight-conversion strategy, latent metadata, a weight source, and an optional mflux
live-preview binding. Adding a variant means adding a kernel entry and a small public class. You shouldn't need to
thread changes through `api.py` or `convert.py`. Two hard requirements: the parity suite stays
green bit-for-bit, and the runtime wheel must not gain a PyTorch dependency (torch is dev-only,
for generating fixtures).

## Examples

Scripts under `examples/` ship inside the published package, so people read them as
authoritative usage. Keep them lint-clean, and make sure the comments and docstrings match
what the code actually does — a stale comment in an example misleads more than no comment.

## Performance claims

If a change claims a speedup, back it with a reproducible benchmark committed to the repo —
pinned seed, prompt, dimensions, step count, and dtype, several timed runs, and the median
reported (not a single best-case number). One-off measurements printed by an example script
are fine for illustration, but the numbers that land in the README or benchmark tables come
from the committed harness.

## Pull requests

- Fork the repo, create a branch, and open your PR against `main`.
- Keep each PR focused on one change — it's easier to review and faster to merge.
- Get the checks above green locally first; CI runs them on every PR.
- Write clear, descriptive commit messages.

A maintainer will review and merge. Thanks for contributing.
