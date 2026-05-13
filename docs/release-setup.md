# Release setup (one-time, by repo owner)

The `release.yml` workflow uses **PyPI Trusted Publishing (OIDC)** — no
long-lived API tokens needed. To activate it, the repo owner does this once:

## 1. PyPI Pending Publisher

1. Log into https://pypi.org (must own the `mlx-taef` project).
2. Account settings → Publishing → "Add a new pending publisher"
   (or under the project: "Manage" → "Publishing").
3. Fill in exactly:
   - **PyPI Project Name:** `mlx-taef`
   - **Owner:** `IonDen`
   - **Repository name:** `mlx-taef`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
4. Save.

## 2. GitHub `pypi` environment

Already created at https://github.com/IonDen/mlx-taef/settings/environments/pypi.

Optional: add a required reviewer under "Deployment protection rules" to
require a human approval before the workflow uploads. Recommended once
releases are user-facing.

## 3. Trigger a release

```bash
# From a clean main:
git tag -a v0.1.1 -m "v0.1.1"
git push --tags
```

Watch the Release workflow run. If the Pending Publisher is configured,
the upload succeeds without any token. GitHub will also create a release
with auto-generated notes from PR/commit history since the previous tag.

Tags that contain a hyphen (e.g. `v0.2.0-alpha`) are automatically marked
as prerelease on GitHub.

## Why Trusted Publishing instead of an API token

- No long-lived secret stored in GitHub Secrets that could leak.
- PyPI verifies exactly which workflow it accepts uploads from.
- Token compromise risk is bounded to the duration of one workflow run.

## Emergency manual publish (when CI is broken)

```bash
uv build
UV_PUBLISH_TOKEN='<token>' uv publish
# Revoke the token on pypi.org immediately after.
```
