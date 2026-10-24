import builtins
import importlib.metadata
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest
from packaging.requirements import Requirement

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

# Oldest mflux minor CI exercises (the minor the latest release names as supported); bump it
# when that moves.
MFLUX_FLOOR_MINOR = "0.19"

# The uv every CI job installs before its frozen sync; the version the lockfile was produced
# with. Bump it together with the `pip install uv==` lines in ci.yml.
UV_VERSION = "0.11.2"

# The four mflux modules the floor guard must import, so a floor mflux that fails to import
# turns the job red instead of letting the suite's `pytest.importorskip` skip the mflux tests.
_MFLUX_GUARD_IMPORTS = (
    "mflux",
    "mflux.callbacks.callback_registry",
    "mflux.callbacks.generation_context",
    "mflux.models.common.config.model_config",
)

_MFLUX_FLOOR_JOB = "test-mflux-floor"
_GUARD_OPENING = "uv run --no-sync python -c '"


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def _job_block(job_id: str) -> list[str]:
    """The raw lines of one ci.yml job after its `  <id>:` key, blank or whitespace-only lines
    skipped, up to the first non-blank line indented two spaces or less (the next job key, or
    a top-level key after the last job)."""
    lines = _CI_WORKFLOW.read_text().splitlines()
    header = f"  {job_id}:"
    assert header in lines, f"ci.yml has no `{job_id}` job"
    block: list[str] = []
    for line in lines[lines.index(header) + 1 :]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip(" ")) <= 2:
            break
        block.append(line)
    return block


def _job_ids() -> list[str]:
    """The ids of the ci.yml jobs: the keys indented two spaces under the top-level `jobs:`."""
    lines = _CI_WORKFLOW.read_text().splitlines()
    assert "jobs:" in lines, "ci.yml has no top-level `jobs:`"
    ids: list[str] = []
    for line in lines[lines.index("jobs:") + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break
        if len(line) - len(line.lstrip(" ")) == 2:
            ids.append(line.strip().removesuffix(":"))
    return ids


def _top_level_block(key: str) -> list[str]:
    """The stripped lines under every unindented `<key>:` line of ci.yml, up to the next
    unindented line, blank and `#` lines skipped; a value on the key's own line comes first
    (so `permissions: write-all` reads as `["write-all"]`)."""
    block: list[str] = []
    inside = False
    for line in _CI_WORKFLOW.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            inside = line.split(":", 1)[0] == key
            value = line.split(":", 1)[1].strip() if inside else ""
            if value:
                block.append(value)
            continue
        if inside:
            block.append(line.strip())
    return block


def _job_lines(job_id: str) -> list[str]:
    """The stripped lines of one ci.yml job (bounded as in `_job_block`), without lines that
    start with `#`. The filter is textual, so `#` lines inside a `run: |` block (shell or
    Python comments) are dropped too."""
    return [line.strip() for line in _job_block(job_id) if not line.strip().startswith("#")]


def _job_steps(job_id: str) -> list[list[str]]:
    """`_job_lines` grouped into steps: each step starts at a line beginning with `- `."""
    steps: list[list[str]] = []
    for line in _job_lines(job_id):
        if line.startswith("- "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    return steps


def _runs_on(job_id: str) -> str:
    runs_on = [line for line in _job_lines(job_id) if line.startswith("runs-on:")]
    assert len(runs_on) == 1, runs_on
    return runs_on[0]


def _floor_guard() -> tuple[str, str]:
    """The floor job's guard program (the lines between the `python -c '` line and the first
    line that starts with `'`, dedented) and that closing line, stripped."""
    block = _job_block(_MFLUX_FLOOR_JOB)
    openings = [i for i, line in enumerate(block) if line.rstrip().endswith("python -c '")]
    assert len(openings) == 1, f"expected one `python -c '` line in the job, got {openings}"
    rest = block[openings[0] + 1 :]
    closings = [i for i, line in enumerate(rest) if line.strip().startswith("'")]
    assert closings, "the guard program never closes its quote"
    return textwrap.dedent("\n".join(rest[: closings[0]])), rest[closings[0]].strip()


class _StubModule(types.ModuleType):
    """Stands in for an mflux module; any attribute is another stub."""

    def __getattr__(self, name: str) -> "_StubModule":
        return _StubModule(f"{self.__name__}.{name}")


def _run_floor_guard(monkeypatch: pytest.MonkeyPatch, mflux_version: str) -> list[str]:
    """Execute the guard program as `python -c` would, with `importlib.metadata.version`
    reporting `mflux_version` for mflux and 0.32.2 for mlx. Imports of `mflux` and its
    submodules get stubs instead of the installed mflux, and their names are returned in
    the order the program requested them; every other import is real."""
    program, _ = _floor_guard()
    installed = {"mflux": mflux_version, "mlx": "0.32.2"}

    def fake_version(distribution: str) -> str:
        if distribution not in installed:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return installed[distribution]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    requested: list[str] = []
    real_import = builtins.__import__

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "mflux" or name.startswith("mflux.")):
            requested.append(name)
            requested.extend(f"{name}.{item}" for item in fromlist or ())
            return _StubModule(name if fromlist else "mflux")
        return real_import(name, globals, locals, fromlist, level)

    namespace = {
        "__name__": "__main__",
        "__builtins__": {**vars(builtins), "__import__": recording_import},
    }
    exec(compile(program, "<ci.yml floor guard>", "exec"), namespace)
    return requested


def _floor_version(minor_offset: int, patch: int) -> str:
    """An mflux version `minor_offset` minors away from the floor minor, e.g. (1, 0) -> 0.20.0
    while the floor is 0.19."""
    major, minor = MFLUX_FLOOR_MINOR.split(".")
    return f"{major}.{int(minor) + minor_offset}.{patch}"


def _dependency_places(pyproject: dict) -> dict[str, list]:
    """Every dependency list in pyproject keyed by where it sits: `extra <name>` for each
    `[project.optional-dependencies]` entry, `group <name>` for each `[dependency-groups]` one."""
    return {
        **{
            f"extra {name}": deps
            for name, deps in pyproject["project"]["optional-dependencies"].items()
        },
        **{f"group {name}": deps for name, deps in pyproject["dependency-groups"].items()},
    }


def _mflux_requirements(pyproject: dict) -> dict[str, Requirement]:
    """Every `mflux` requirement in pyproject, keyed by where it sits: each extra under
    `[project.optional-dependencies]` and each `[dependency-groups]` group (table entries such
    as `{include-group = ...}` are skipped)."""
    found: dict[str, Requirement] = {}
    for place, deps in _dependency_places(pyproject).items():
        for dep in deps:
            if isinstance(dep, str) and Requirement(dep).name == "mflux":
                assert place not in found, f"two mflux requirements in {place}"
                found[place] = Requirement(dep)
    return found


def test_every_mflux_requirement_starts_at_the_ci_floor_minor() -> None:
    """The floor job installs `mflux=={MFLUX_FLOOR_MINOR}.*`, and the install metadata must
    admit nothing older, or a user resolves an mflux no CI job has run. The one-line bug this
    catches: moving the lower bound of one mflux specifier (the `mflux` or `showcase` extra,
    or the `test` group) without the others or without `MFLUX_FLOOR_MINOR`, e.g. leaving one
    at `>=0.17`."""
    found = _mflux_requirements(_pyproject())

    assert set(found) == {"extra mflux", "extra showcase", "group test"}, sorted(found)
    for place, requirement in found.items():
        lower = [spec for spec in requirement.specifier if spec.operator == ">="]
        assert [spec.version for spec in lower] == [MFLUX_FLOOR_MINOR], (place, str(requirement))


def test_built_distributions_declare_core_metadata_2_4() -> None:
    """The v0.8.1 tag push failed to publish: hatchling 1.32 emits Metadata-Version 2.5 and
    the release workflow's `twine check --strict` (twine 6.2.0, bundled by the build action)
    rejects it. Pinning the emitted core-metadata version to 2.4 keeps the wheel and sdist
    readable by every tool in the release path. The one-line bug this catches: dropping
    `core-metadata-version` from either hatch build target."""
    targets = _pyproject()["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["core-metadata-version"] == "2.4"
    assert targets["sdist"]["core-metadata-version"] == "2.4"


def test_ci_lint_job_checks_a_built_wheel_with_the_release_twine() -> None:
    """The metadata/tooling mismatch was only ever exercised at tag time. The lint job must
    build the distributions and run the same `twine check --strict` the release action runs,
    with twine pinned to the version that action bundles, so the mismatch fails a pull
    request instead of a release. The one-line bug this catches: removing the check step or
    unpinning twine to whatever is newest (which would pass while the release tooling fails)."""
    workflow = _CI_WORKFLOW.read_text()
    assert "run: uv build" in workflow
    check_lines = [
        line.strip() for line in workflow.splitlines() if "twine" in line and "check" in line
    ]
    assert len(check_lines) == 1
    assert "twine==6.2.0" in check_lines[0]
    assert "check --strict dist/*" in check_lines[0]


def test_ci_uv_sync_commands_are_frozen() -> None:
    workflow = _CI_WORKFLOW.read_text()
    sync_commands = [line.strip() for line in workflow.splitlines() if "run: uv sync" in line]

    assert len(sync_commands) == 4
    assert all("--frozen" in command for command in sync_commands)


def test_ci_every_job_installs_the_pinned_uv() -> None:
    """The frozen sync only pins what uv installs, not uv itself: an unpinned `pip install uv`
    lets a new uv release change how the lockfile is read under every job. The one-line bug
    this catches: dropping or changing the `==UV_VERSION` pin on any job's `pip install uv`
    line, or a job that installs uv twice."""
    jobs = _job_ids()
    workflow_installs = [
        line.strip() for line in _CI_WORKFLOW.read_text().splitlines() if "pip install uv" in line
    ]

    assert jobs, "ci.yml declares no jobs"
    for job in jobs:
        installs = [line for line in _job_lines(job) if "pip install uv" in line]
        assert installs == [f"run: pip install uv=={UV_VERSION}"], (job, installs)
    assert len(workflow_installs) == len(jobs), workflow_installs


def test_ci_workflow_token_is_read_only() -> None:
    """Without a top-level `permissions:` the token's scope comes from a repository setting,
    not from the file, and a job-level block replaces the top-level one wholesale. The
    one-line bug this catches: deleting the top-level block, widening it (`contents: write`,
    `write-all`), adding a second scope, or a job-level `permissions:` such as
    `permissions: write-all`."""
    assert _top_level_block("permissions") == ["contents: read"]
    for job in _job_ids():
        overrides = [
            line
            for line in _job_lines(job)
            if line.removeprefix("- ").split(":", 1)[0].strip() == "permissions"
        ]
        assert not overrides, (job, overrides)


def test_ci_checkouts_do_not_persist_the_token() -> None:
    """`actions/checkout` writes the job token into `.git/config` unless told not to, where any
    later step can read it; no step here needs git credentials. The one-line bug this
    catches: removing `persist-credentials: false` (or setting it to true) on any job's
    checkout, or a checkout step this test does not see."""
    checkouts = {
        job: [step for step in _job_steps(job) if "actions/checkout@" in step[0]]
        for job in _job_ids()
    }
    workflow_checkouts = [
        line for line in _CI_WORKFLOW.read_text().splitlines() if "actions/checkout@" in line
    ]

    assert all(checkouts.values()), checkouts
    for job, steps in checkouts.items():
        for step in steps:
            assert "with:" in step, (job, step)
            assert "persist-credentials: false" in step, (job, step)
    assert len(workflow_checkouts) == sum(len(steps) for steps in checkouts.values())


def test_every_place_that_requires_mflux_caps_pillow_below_13() -> None:
    """mflux 0.23.0 fails at `import mflux` with Pillow 13 (mflux #890, fixed on mflux main
    but not released). Every extra or group that pulls in mflux (the `mflux` and `showcase`
    extras, the `test` group) must carry its own Pillow cap: an install resolves only the
    place it names. The one-line bug this catches: `pip install mlx-taef[showcase]` (or a CI
    sync of the `test` group) resolving Pillow 13 and crashing at `import mflux`, because the
    cap sits on one place only or moved off `<13`."""
    data = _pyproject()
    places = _dependency_places(data)
    mflux_places = set(_mflux_requirements(data))

    assert mflux_places == {"extra mflux", "extra showcase", "group test"}, sorted(mflux_places)
    for place in sorted(mflux_places):
        pillow = [
            Requirement(dep)
            for dep in places[place]
            if isinstance(dep, str) and Requirement(dep).name.lower() == "pillow"
        ]
        assert len(pillow) == 1, (place, places[place])
        upper = [spec.version for spec in pillow[0].specifier if spec.operator == "<"]
        assert upper == ["13"], (place, str(pillow[0]))
        assert pillow[0].marker is None, (place, str(pillow[0]))


def test_job_lines_reads_the_whole_job_and_nothing_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The job tests below look only at what `_job_lines` returns, so a job cut short would
    hide its later lines from them (a `continue-on-error` after a stray indented blank line
    would pass unseen). The one-line bug this catches: stopping the job at a whitespace-only
    line, or reading on past the job into the next job's key or a top-level key that follows."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "jobs:\n"
        "  a:\n"
        "    runs-on: x\n"
        "  \n"
        "\n"
        "    steps:\n"
        "      # a comment\n"
        "      - run: echo hi\n"
        "  b:\n"
        "    runs-on: y\n"
        "env:\n"
        "  KEY: value\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_CI_WORKFLOW", workflow)

    assert _job_lines("a") == ["runs-on: x", "steps:", "- run: echo hi"]


def test_job_ids_and_top_level_block_read_what_the_hardening_tests_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The token and uv tests see only the jobs `_job_ids` lists and the scopes
    `_top_level_block` returns. The one-line bug this catches: `_job_ids` listing step keys or
    missing a job after a comment, or `_top_level_block` dropping a value written on the key's
    own line (so `permissions: write-all` would read as no block) or reading on past the next
    top-level key."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "permissions: write-all\n"
        "  contents: read\n"
        "jobs:\n"
        "  a:\n"
        "    runs-on: x\n"
        "  # a comment\n"
        "  b:\n"
        "    steps:\n"
        "      - run: echo hi\n"
        "env:\n"
        "  KEY: value\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_CI_WORKFLOW", workflow)

    assert _job_ids() == ["a", "b"]
    assert _top_level_block("permissions") == ["write-all", "contents: read"]


def test_ci_mflux_floor_job_installs_the_floor_after_a_frozen_sync() -> None:
    """The lockfile resolves mflux to the newest minor it admits on macOS, so 0.19, the oldest
    minor the release names as supported (`MFLUX_FLOOR_MINOR`), is only exercised by a job
    that installs it over the frozen test sync. The one-line bug
    this catches: deleting the job, or installing the floor before the sync (which would
    put the locked mflux straight back)."""
    job = _job_lines(_MFLUX_FLOOR_JOB)

    sync = job.index("run: uv sync --frozen --group test")
    install = job.index(f'run: uv pip install --no-deps "mflux=={MFLUX_FLOOR_MINOR}.*"')
    assert sync < install


def test_ci_mflux_floor_job_runs_on_the_test_jobs_runner() -> None:
    """The floor leg is the `test` matrix's suite with one older mflux; on another runner
    image a failure or a pass could come from the platform instead of mflux. The one-line
    bug this catches: moving the `test` job to a new runner and leaving the floor job on the
    old one, or the reverse."""
    assert _runs_on(_MFLUX_FLOOR_JOB) == _runs_on("test")


def test_ci_mflux_floor_install_leaves_the_locked_dependencies_alone() -> None:
    """A plain `uv pip install` re-resolves the new mflux's dependencies live, so a release
    or a lock move could change mlx (or anything else) under the floor leg. The one-line bug
    this catches: dropping `--no-deps` from the install."""
    installs = [line for line in _job_lines(_MFLUX_FLOOR_JOB) if "uv pip install" in line]

    assert installs, "the floor job installs no mflux"
    assert all("--no-deps" in line for line in installs), installs


def test_ci_mflux_floor_job_installs_once_and_runs_pytest_right_after_the_guard() -> None:
    """The guard only vouches for the venv pytest sees if nothing changes that venv between
    the two steps. The one-line bug this catches: a second `uv sync` or `uv pip install` in
    the job (after the guard it would swap mflux back to the locked or a newer one), or any
    step inserted between the guard and pytest."""
    job = _job_lines(_MFLUX_FLOOR_JOB)
    steps = _job_steps(_MFLUX_FLOOR_JOB)
    guard = [i for i, step in enumerate(steps) if _GUARD_OPENING in step]

    assert len([line for line in job if "uv sync" in line]) == 1, job
    assert len([line for line in job if "uv pip install" in line]) == 1, job
    assert len(guard) == 1, steps
    after_guard = steps[guard[0] + 1 : guard[0] + 2]
    assert after_guard, "nothing runs after the guard"
    assert any("run: uv run --no-sync pytest" in line for line in after_guard[0]), after_guard


def test_ci_mflux_floor_job_cannot_be_switched_off() -> None:
    """Each of these lets the floor job report green without testing the floor. The one-line
    bug this catches: `continue-on-error: true` or an `if:` (e.g. `if: false`) on the job or one of
    its steps, or a `PYTEST_ADDOPTS` env in the job that rewrites the pytest selection."""
    job = _job_lines(_MFLUX_FLOOR_JOB)
    keys = [line.removeprefix("- ").split(":", 1)[0].strip() for line in job]

    assert not any("continue-on-error" in line for line in job), job
    assert "if" not in keys, job
    assert not any("PYTEST_ADDOPTS" in line for line in job), job


def test_ci_mflux_floor_job_comment_names_the_floor_the_extras_admit() -> None:
    """The floor job's comment tells a reader which mflux the job covers and what the pins
    admit. The one-line bug this catches: the comment still claiming the pins admit a minor
    below the floor (0.17, 0.18) after the extras moved to `>=0.19`, or losing the floor."""
    comment = " ".join(
        line.strip().removeprefix("#").strip()
        for line in _job_block(_MFLUX_FLOOR_JOB)
        if line.strip().startswith("#")
    )
    floor = int(MFLUX_FLOOR_MINOR.split(".")[1])
    minors = [int(m) for m in re.findall(r"\b0\.(\d+)\b", comment)]

    assert f"(`>={MFLUX_FLOOR_MINOR}`)" in comment, comment
    assert minors, comment
    assert all(minor >= floor for minor in minors), comment


def test_ci_mflux_floor_job_never_lets_uv_run_resync() -> None:
    """Plain `uv run` re-syncs the venv to the lockfile and silently swaps the floor mflux
    back to the locked one, so the job tests the newest minor twice and stays green. The
    one-line bug this catches: dropping `--no-sync` from any `uv run` in the job."""
    uv_runs = [line for line in _job_lines(_MFLUX_FLOOR_JOB) if "uv run" in line]

    assert len(uv_runs) >= 2, uv_runs
    assert all("uv run --no-sync " in line for line in uv_runs), uv_runs


def test_ci_mflux_floor_guard_is_the_whole_step_script() -> None:
    """The guard fails the job only through its exit status, so its step's script must be
    the guard command alone, ending on the quote that closes the program. The one-line bug
    this catches: `' || true` (or any other text) after the closing quote, or another
    command before or after the guard in the same step."""
    _, closing = _floor_guard()
    guard_steps = [step for step in _job_steps(_MFLUX_FLOOR_JOB) if _GUARD_OPENING in step]

    assert closing == "'"
    assert len(guard_steps) == 1, guard_steps
    step = guard_steps[0]
    assert step[step.index(_GUARD_OPENING) - 1] == "run: |", step
    assert step[-1] == "'", step


def test_ci_mflux_floor_guard_accepts_a_floor_patch_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must let a real floor run through to pytest. The one-line bug this catches:
    a guard that exits (or crashes) on a floor patch release such as 0.19.1, e.g. an
    unconditional `raise SystemExit(0)`, which skips the suite with a green step."""
    try:
        _run_floor_guard(monkeypatch, _floor_version(0, 1))
    except SystemExit as exit_:
        pytest.fail(f"the guard exited with code {exit_.code!r} on the floor minor")


@pytest.mark.parametrize(
    "mflux_version",
    [_floor_version(1, 0), _floor_version(-1, 2)],
    ids=["next-minor", "previous-minor"],
)
def test_ci_mflux_floor_guard_fails_the_job_off_the_floor_minor(
    monkeypatch: pytest.MonkeyPatch, mflux_version: str
) -> None:
    """A resolver or tooling change that leaves another mflux in place must not look like a
    passing floor run: off the floor minor the guard has to exit non-zero (an int code other
    than 0, or a message, which Python exits with 1). The one-line bug this catches: a guard
    that also accepts the next or the previous minor (e.g. `or mflux.startswith("0.20.")`),
    a check that never fires, or `raise SystemExit(0)` instead of a failing exit."""
    with pytest.raises(SystemExit) as exit_:
        _run_floor_guard(monkeypatch, mflux_version)

    code = exit_.value.code
    assert isinstance(code, str) or (isinstance(code, int) and code != 0), code


def test_ci_mflux_floor_guard_imports_the_mflux_modules_the_suite_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mflux tests use `pytest.importorskip`, so a floor mflux that fails to import
    would skip them and leave the job green; only the guard importing them turns that red.
    The one-line bug this catches: dropping one of the four mflux imports from the guard, or
    moving it to a branch a floor run never executes."""
    requested = _run_floor_guard(monkeypatch, _floor_version(0, 1))

    missing = [module for module in _MFLUX_GUARD_IMPORTS if module not in requested]
    assert not missing, f"the guard never imports {missing}; it requested {requested}"


def test_ci_mflux_floor_job_runs_the_offline_suite_without_a_coverage_gate() -> None:
    """The floor leg must run the same offline selection as the matrix, but a single older
    mflux may legitimately cover different lines, so the 95% gate stays on the matrix. The
    one-line bug this catches: a marker expression that skips the suite (or runs network
    tests) or a copied `--cov-fail-under` that fails the leg for coverage, not behaviour."""
    job = _job_lines(_MFLUX_FLOOR_JOB)
    pytest_runs = [line for line in job if "pytest" in line]

    assert pytest_runs == ['run: uv run --no-sync pytest -m "not benchmark and not network"']
    assert not any("--cov-fail-under" in line for line in job)


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


def test_mlx_teacache_pins_start_at_0_14_0_below_1_on_every_python() -> None:
    """Both mlx-teacache pins (showcase extra, test group) read exactly `>=0.14.0,<1` with no
    Python marker: a 3.11 marker drops it on 3.10, a stale `<0.10` cap locks a pre-0.14
    release, and a floor moved anywhere off 0.14.0 (down to 0.13.5, which a containment check
    against 0.13.4 lets through, or up past 0.14.0) changes what the combined showcase runs."""
    data = _pyproject()
    entries = [
        Requirement(dep)
        for dep in (
            data["project"]["optional-dependencies"]["showcase"] + data["dependency-groups"]["test"]
        )
        if isinstance(dep, str) and dep.startswith("mlx-teacache")
    ]
    assert len(entries) == 2, entries
    for req in entries:
        assert req.marker is None, f"stale python gate: {req}"
        assert req.specifier.contains("0.14.0"), f"must admit 0.14.0: {req}"
        assert req.specifier.contains("0.99.0"), f"must admit later 0.x releases: {req}"
        assert not req.specifier.contains("0.13.4"), f"floor must exclude 0.13.x: {req}"
        assert not req.specifier.contains("1.0.0"), f"must cap below 1.0: {req}"
        lower = [spec.version for spec in req.specifier if spec.operator == ">="]
        upper = [spec.version for spec in req.specifier if spec.operator == "<"]
        assert lower == ["0.14.0"], f"floor must be exactly 0.14.0: {req}"
        assert upper == ["1"], f"cap must be exactly <1: {req}"
        assert len(req.specifier) == 2, f"only the floor and the cap: {req}"
