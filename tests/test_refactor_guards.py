"""Guard: the v0.3.0 kernel refactor is zero-behavior-change vs the base branch.

It must not modify committed reference fixtures, converted weights, or the fixture manifest —
those are the bit-exact parity oracle. A diff against the merge-base catches any such change.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROTECTED = ("tests/reference", "tests/converted", "tests/fixtures.toml")


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


def test_phase1_does_not_modify_committed_fixtures():
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
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}...HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    touched = [f for f in changed if any(f.startswith(p) for p in PROTECTED)]
    assert touched == [], f"refactor must not modify committed fixtures, but changed: {touched}"
