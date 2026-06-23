"""Guard: the committed parity oracle must not be silently MODIFIED or DELETED.

Committed reference fixtures and converted weights are the bit-exact oracle. A diff against the
merge-base catches any modification/deletion of an existing oracle file. New model fixtures may be
ADDED (e.g. a new variant's reference latents + converted weights) — only changes to files that
already existed are rejected. `tests/fixtures.toml` is covered separately by
`test_fixtures_integrity` (every listed hash must match), so it is not guarded here.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROTECTED = ("tests/reference", "tests/converted")


def _base_ref() -> str | None:
    for ref in ("origin/main", "main"):
        r = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", ref],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return ref
    return None


def test_committed_fixtures_are_not_modified_or_deleted():
    base = _base_ref()
    if base is None:
        pytest.skip("no origin/main or main ref to diff against")
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # --diff-filter=MD: only Modified/Deleted files — Added (new-model) fixtures are allowed.
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=MD", f"{merge_base}...HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    touched = [f for f in changed if any(f.startswith(p) for p in PROTECTED)]
    assert touched == [], (
        f"committed oracle fixtures must not be modified/deleted, changed: {touched}"
    )
