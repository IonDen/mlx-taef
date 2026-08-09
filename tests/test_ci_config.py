import sys
from pathlib import Path

# Version-gated import: no single CI leg runs both arms, so both are excluded
# from coverage rather than reporting one as untested on every interpreter.
if sys.version_info >= (3, 11):  # pragma: no cover
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

_REPO_ROOT = Path(__file__).parent.parent
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The single source of truth for "which Pythons do we claim to support".
# Every surface below must agree with it, so a widened floor can never ship
# untested (the CI matrix entry IS the test for a version).
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def test_ci_uv_sync_commands_are_frozen() -> None:
    workflow = _CI_WORKFLOW.read_text()
    sync_commands = [line.strip() for line in workflow.splitlines() if "run: uv sync" in line]

    assert len(sync_commands) == 3
    assert all("--frozen" in command for command in sync_commands)


def test_requires_python_floor_matches_lowest_supported_version() -> None:
    assert _pyproject()["project"]["requires-python"] == f">={SUPPORTED_PYTHONS[0]}"


def test_classifiers_list_every_supported_version() -> None:
    classifiers = _pyproject()["project"]["classifiers"]
    declared = {
        c.rsplit(" :: ", 1)[1]
        for c in classifiers
        if c.startswith("Programming Language :: Python :: ") and c[-1].isdigit() and "." in c
    }

    assert declared == set(SUPPORTED_PYTHONS)


def test_ci_test_matrix_covers_every_supported_version() -> None:
    """A version we advertise but never run is an untested claim."""
    workflow = _CI_WORKFLOW.read_text()
    matrix_lines = [line for line in workflow.splitlines() if "python:" in line and "[" in line]

    assert len(matrix_lines) == 1, matrix_lines
    matrix = {
        part.strip().strip('"') for part in matrix_lines[0].split("[")[1].rstrip("]").split(",")
    }

    assert matrix == set(SUPPORTED_PYTHONS)


def test_mypy_targets_the_lowest_supported_version() -> None:
    """Type-checking the floor is what catches typing that only works on newer Pythons."""
    assert _pyproject()["tool"]["mypy"]["python_version"] == SUPPORTED_PYTHONS[0]


def test_readme_python_badge_states_the_floor_as_a_range() -> None:
    """The badge is hand-written (shields' pyversions enumerates every version), so it
    must be pinned to the declared floor or it will quietly advertise the wrong one."""
    readme = (_REPO_ROOT / "README.md").read_text()
    floor = SUPPORTED_PYTHONS[0]

    assert f"badge/python-{floor}%2B-" in readme, (
        f"README needs a python-{floor}%2B badge matching requires-python"
    )
    assert "pypi/pyversions" not in readme, (
        "the enumerating pyversions badge was replaced by the floor badge"
    )


def test_mlx_teacache_has_no_python_version_marker() -> None:
    """mlx-teacache supports 3.10 since its v0.9.3; the old gate must not silently return."""
    data = _pyproject()
    entries = [
        dep
        for dep in (
            data["project"]["optional-dependencies"]["showcase"]
            + data["dependency-groups"]["test"]
        )
        if isinstance(dep, str) and dep.startswith("mlx-teacache")
    ]
    assert len(entries) == 2, entries
    for dep in entries:
        assert "python_version" not in dep, f"stale 3.11 gate: {dep}"
        assert ">=0.9.3" in dep, f"floor must be the first 3.10-capable release: {dep}"
