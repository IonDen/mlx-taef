import builtins
import importlib.metadata
import itertools
import re
import shlex
import sys
import textwrap
import tomllib
import types
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

_REPO_ROOT = Path(__file__).parent.parent
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_MFLUX_LATEST_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "mflux-latest.yml"
_MFLUX_LATEST_JOB = "test-mflux-latest"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# A `uses:` line pinned to a full commit SHA, with the exact release tag of that commit as its
# trailing comment (a floating tag such as `# v6` does not count).
_PINNED_USES = re.compile(r"(- )?uses: [\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+")

# A `uses` key in any spelling YAML accepts: bare, quoted (`"uses":`, `'uses':`) or with
# spaces before the colon (`uses :`).
_USES_KEY = re.compile(r"""\buses['"]?\s*:""")

# The single source of truth for "which Pythons do we claim to support".
# Every surface below must agree with it, so a widened floor can never ship
# untested (the CI matrix entry IS the test for a version).
SUPPORTED_PYTHONS = ("3.11", "3.12", "3.13", "3.14")

# Oldest mflux minor CI exercises (the minor the latest release names as supported); bump it
# when that moves.
MFLUX_FLOOR_MINOR = "0.24"

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


def _job_block(job_id: str, workflow: Path | None = None) -> list[str]:
    """The raw lines of one job of `workflow` (ci.yml when None) after its `  <id>:` key, blank
    or whitespace-only lines skipped, up to the first non-blank line indented two spaces or less
    (the next job key, or a top-level key after the last job)."""
    workflow = workflow or _CI_WORKFLOW
    lines = workflow.read_text().splitlines()
    header = f"  {job_id}:"
    assert header in lines, f"{workflow.name} has no `{job_id}` job"
    block: list[str] = []
    for line in lines[lines.index(header) + 1 :]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip(" ")) <= 2:
            break
        block.append(line)
    return block


def _job_ids(workflow: Path | None = None) -> list[str]:
    """The ids of the jobs of `workflow` (ci.yml when None): the keys indented two spaces under
    the top-level `jobs:`."""
    workflow = workflow or _CI_WORKFLOW
    lines = workflow.read_text().splitlines()
    assert "jobs:" in lines, f"{workflow.name} has no top-level `jobs:`"
    ids: list[str] = []
    for line in lines[lines.index("jobs:") + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break
        if len(line) - len(line.lstrip(" ")) == 2:
            ids.append(line.strip().removesuffix(":"))
    return ids


def _top_level_block(key: str, workflow: Path | None = None) -> list[str]:
    """The stripped lines under every unindented `<key>:` line of `workflow` (ci.yml when None),
    up to the next unindented line, blank and `#` lines skipped; a value on the key's own line
    comes first (so `permissions: write-all` reads as `["write-all"]`)."""
    block: list[str] = []
    inside = False
    for line in (workflow or _CI_WORKFLOW).read_text().splitlines():
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


def _job_lines(job_id: str, workflow: Path | None = None) -> list[str]:
    """The stripped lines of one job (bounded as in `_job_block`), without lines that start
    with `#`. The filter is textual, so `#` lines inside a `run: |` block (shell or Python
    comments) are dropped too."""
    return [
        line.strip() for line in _job_block(job_id, workflow) if not line.strip().startswith("#")
    ]


def _job_steps(job_id: str, workflow: Path | None = None) -> list[list[str]]:
    """`_job_lines` grouped into steps: each step starts at a line beginning with `- `."""
    steps: list[list[str]] = []
    for line in _job_lines(job_id, workflow):
        if line.startswith("- "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    return steps


def _job_permissions(job_id: str, workflow: Path) -> list[str]:
    """The stripped scope lines of one job's own `permissions:` key (a value on the key's line
    comes first), or [] when the job sets none."""
    block = _job_block(job_id, workflow)
    keys = [i for i, line in enumerate(block) if line.strip().startswith("permissions:")]
    assert len(keys) <= 1, (job_id, keys)
    if not keys:
        return []
    key_line = block[keys[0]]
    indent = len(key_line) - len(key_line.lstrip(" "))
    value = key_line.split(":", 1)[1].strip()
    scopes = [value] if value else []
    for line in block[keys[0] + 1 :]:
        if len(line) - len(line.lstrip(" ")) <= indent:
            break
        scopes.append(line.strip())
    return scopes


def _unpinned_uses(text: str) -> list[str]:
    """The stripped `uses:` lines of a workflow that are not `owner/repo@<40 hex> # vX.Y.Z`.
    Comment lines are skipped; any other line with a `uses` key (also quoted, `"uses":`, or
    spaced, `uses :`) is read as a step."""
    uses = [
        line.strip()
        for line in text.splitlines()
        if _USES_KEY.search(line) and not line.lstrip().startswith("#")
    ]
    return [line for line in uses if not _PINNED_USES.fullmatch(line)]


def _step_indices(steps: list[list[str]], text: str) -> list[int]:
    """The indices of the steps with a line containing `text`."""
    return [i for i, step in enumerate(steps) if any(text in line for line in step)]


def _runs_on(job_id: str) -> str:
    runs_on = [line for line in _job_lines(job_id) if line.startswith("runs-on:")]
    assert len(runs_on) == 1, runs_on
    return runs_on[0]


_HF_HUB_OFFLINE_LINES = {'HF_HUB_OFFLINE:"1"', "HF_HUB_OFFLINE:'1'", "HF_HUB_OFFLINE:1"}


def _sets_hf_hub_offline(step: list[str]) -> bool:
    """Whether a step (stripped lines, as `_job_steps` gives them) has an `env:` key that sets
    `HF_HUB_OFFLINE` to 1."""
    return "env:" in step and any(line.replace(" ", "") in _HF_HUB_OFFLINE_LINES for line in step)


def _pytest_step(job_id: str, workflow: Path | None = None) -> list[str]:
    """The one step of a job with a line that runs pytest."""
    steps = _job_steps(job_id, workflow)
    found = _step_indices(steps, "pytest")
    assert len(found) == 1, (job_id, steps)
    return steps[found[0]]


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


def _inline_programs(job_id: str, workflow: Path) -> list[tuple[int, str, str]]:
    """Every `python -c '` program of a job as (step index, program, closing line): the step
    index counts steps as `_job_steps` does, the program is the lines between the opening line
    and the first line that starts with `'` (dedented), the closing line is stripped."""
    block = [line for line in _job_block(job_id, workflow) if not line.strip().startswith("#")]
    programs: list[tuple[int, str, str]] = []
    step = -1
    for i, line in enumerate(block):
        if line.strip().startswith("- "):
            step += 1
        if line.rstrip().endswith("python -c '"):
            rest = block[i + 1 :]
            closings = [j for j, item in enumerate(rest) if item.strip().startswith("'")]
            assert closings, f"a `python -c '` program in {job_id} never closes its quote"
            program = textwrap.dedent("\n".join(rest[: closings[0]]))
            programs.append((step, program, rest[closings[0]].strip()))
    return programs


def _run_floor_guard(
    monkeypatch: pytest.MonkeyPatch, mflux_version: str, pillow_version: str = "13.0.0"
) -> list[str]:
    """Execute the floor guard program as `python -c` would; see `_exec_program`."""
    program, _ = _floor_guard()
    return _exec_program(
        monkeypatch, program, mflux_version, pillow_version, "<ci.yml floor guard>"
    )


def _exec_program(
    monkeypatch: pytest.MonkeyPatch,
    program: str,
    mflux_version: str,
    pillow_version: str,
    filename: str,
    failing: str | None = None,
) -> list[str]:
    """Execute a workflow's inline Python program as `python -c` would, with
    `importlib.metadata.version` reporting `mflux_version` for mflux, `pillow_version` for
    pillow and 0.32.2 for mlx. Imports of `mflux` and its submodules get stubs instead of the
    installed mflux, and their names are returned in the order the program requested them;
    every other import is real. With `failing`, importing that one mflux module raises
    ModuleNotFoundError, as a moved or broken module would."""
    installed = {"mflux": mflux_version, "mlx": "0.32.2", "pillow": pillow_version}

    def fake_version(distribution: str) -> str:
        if distribution.lower() not in installed:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return installed[distribution.lower()]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    requested: list[str] = []
    real_import = builtins.__import__

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "mflux" or name.startswith("mflux.")):
            if name == failing:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            requested.append(name)
            requested.extend(f"{name}.{item}" for item in fromlist or ())
            return _StubModule(name if fromlist else "mflux")
        return real_import(name, globals, locals, fromlist, level)

    namespace = {
        "__name__": "__main__",
        "__builtins__": {**vars(builtins), "__import__": recording_import},
    }
    exec(compile(program, filename, "exec"), namespace)
    return requested


# A `0.N` read as an mflux minor: after the word `mflux`, the verb `admit`/`admits`, or a
# version operator (`<`, `>`, `<=`, `>=`, `==`, `!=`, `~=`), followed by any list continuation
# (`, 0.N`, `and 0.N`, `or 0.N`). Each item may carry a patch (`0.17.1`); a continuation item
# followed by a unit of seconds (`s`, `sec`, `secs`, `second`, `seconds`: "and 0.5 s") is a
# duration and ends the list.
_MFLUX_MINOR_MENTION = re.compile(
    r"(?:\b(?:mflux|admits?)\s+|(?:[<>]=?|[=!~]=)\s*)0\.(\d+)(?:\.\d+)?\b"
    r"((?:\s*(?:,|\band\b|\bor\b)\s*0\.\d+(?:\.\d+)?\b(?!\s*(?:s|secs?|seconds?)\b))*)"
)


def _mflux_minors_named_in(comment: str) -> list[int]:
    """The mflux minors a workflow comment names, in order: a `0.N` right after the word
    `mflux`, after `admit`/`admits`, or after a version operator (`mflux 0.19`, `admit 0.19`,
    `>=0.19` and `<0.19` all read as 19), plus every `0.N` of a list that continues it
    (`mflux 0.17, 0.18`, `admit 0.17 and 0.18`, `>=0.17 or 0.18`, `mflux 0.17.1, 0.18`).

    Deliberately not read: a bare `0.N` with none of those in front of it and not inside such
    a list, or a list item followed by a unit of seconds (`s`, `sec`, `secs`, `second`,
    `seconds`: "mflux 0.19 and 0.5 s"), so a duration or ratio in the same prose ("takes 0.5 s")
    is not taken for a version. A comment that names a stale minor only that way ("0.17 is
    still allowed") is not seen."""
    minors: list[int] = []
    for mention in _MFLUX_MINOR_MENTION.finditer(comment):
        minors.append(int(mention.group(1)))
        minors.extend(int(m) for m in re.findall(r"0\.(\d+)(?:\.\d+)?", mention.group(2)))
    return minors


def _floor_version(minor_offset: int, patch: int) -> str:
    """An mflux version `minor_offset` minors away from the floor minor, e.g. (1, 0) -> 0.25.0
    while the floor is 0.24."""
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
    at `>=0.19`."""
    found = _mflux_requirements(_pyproject())

    assert set(found) == {"extra mflux", "extra showcase", "group test"}, sorted(found)
    for place, requirement in found.items():
        lower = [spec for spec in requirement.specifier if spec.operator == ">="]
        assert [spec.version for spec in lower] == [MFLUX_FLOOR_MINOR], (place, str(requirement))


def test_every_mflux_requirement_is_exactly_the_floor_with_no_marker() -> None:
    """mflux 0.24 is the first release that imports on Pillow 13, so every place that installs
    mflux reads exactly `mflux>=0.24`: no ceiling, no exclusions, no environment marker. The
    one-line bug this catches: a leftover `>=0.19`, a cap such as `,<0.26`, or a marker that
    drops mflux on some Python (`; python_version >= "3.12"`) in any of the three places."""
    found = _mflux_requirements(_pyproject())

    assert set(found) == {"extra mflux", "extra showcase", "group test"}, sorted(found)
    for place, requirement in found.items():
        assert str(requirement.specifier) == f">={MFLUX_FLOOR_MINOR}", (place, str(requirement))
        assert requirement.marker is None, (place, str(requirement))


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


def test_release_workflow_token_is_read_only_outside_the_two_publishing_jobs() -> None:
    """Without a top-level `permissions:` the build job's token takes the repository default,
    which can be read-write. Only `publish` (the PyPI Trusted Publishing OIDC token) and
    `github-release` (creating the release) need more, each in its own job. The one-line bug
    this catches: deleting or widening the top-level block, or a job grant moved or widened
    (e.g. `contents: write` on `build`, or `write-all` on `publish`)."""
    grants = {job: _job_permissions(job, _RELEASE_WORKFLOW) for job in _job_ids(_RELEASE_WORKFLOW)}

    assert _top_level_block("permissions", _RELEASE_WORKFLOW) == ["contents: read"]
    assert grants == {
        "build": [],
        "publish": ["id-token: write"],
        "github-release": ["contents: write"],
    }


def test_release_checkouts_do_not_persist_the_token() -> None:
    """The release build only reads the tagged tree (hatch-vcs reads the tags locally), so
    the token has no reason to sit in `.git/config` for the build action's steps. The
    one-line bug this catches: removing `persist-credentials: false` from the build checkout,
    or a checkout step this test does not see."""
    checkouts = [
        step
        for job in _job_ids(_RELEASE_WORKFLOW)
        for step in _job_steps(job, _RELEASE_WORKFLOW)
        if "actions/checkout@" in step[0]
    ]
    workflow_checkouts = [
        line for line in _RELEASE_WORKFLOW.read_text().splitlines() if "actions/checkout@" in line
    ]

    assert checkouts, "release.yml has no checkout"
    assert len(checkouts) == len(workflow_checkouts)
    for step in checkouts:
        assert "persist-credentials: false" in step, step


def test_unpinned_uses_flags_every_pin_but_a_full_sha_with_its_release_tag() -> None:
    """The pin test below is only as strict as `_unpinned_uses`. The one-line bug this
    catches: accepting a tag or branch pin, a short SHA, a SHA with no tag comment, or a
    floating tag (`# v6`) as the comment; or not selecting a `uses` key written quoted
    (`"uses":`) or with a space before the colon (`uses :`), both valid YAML."""
    sha = "d23441a48e516b6c34aea4fa41551a30e30af803"
    workflow = (
        f"      - uses: actions/checkout@{sha} # v6.1.0\n"
        f"        uses: pypa/gh-action-pypi-publish@{sha} # v1.14.3\n"
        "      # uses: actions/checkout@v6\n"
        "      - uses: actions/checkout@v6\n"
        "      - uses: pypa/gh-action-pypi-publish@release/v1\n"
        f"      - uses: actions/checkout@{sha[:12]} # v6.1.0\n"
        f"      - uses: actions/checkout@{sha}\n"
        f"      - uses: actions/checkout@{sha} # v6\n"
        '      - "uses": actions/checkout@v1\n'
        "      - uses : actions/checkout@v1\n"
    )

    assert _unpinned_uses(workflow) == [
        "- uses: actions/checkout@v6",
        "- uses: pypa/gh-action-pypi-publish@release/v1",
        f"- uses: actions/checkout@{sha[:12]} # v6.1.0",
        f"- uses: actions/checkout@{sha}",
        f"- uses: actions/checkout@{sha} # v6",
        '- "uses": actions/checkout@v1',
        "- uses : actions/checkout@v1",
    ]


@pytest.mark.parametrize(
    "workflow",
    [_CI_WORKFLOW, _RELEASE_WORKFLOW, _MFLUX_LATEST_WORKFLOW],
    ids=["ci", "release", "mflux-latest"],
)
def test_every_action_is_pinned_to_a_commit_sha(workflow: Path) -> None:
    """A tag or branch is a moving pointer: whoever controls the action's repository can
    point it at new code that then runs with the job's token. A full commit SHA cannot move.
    The one-line bug this catches: a `uses:` line pinned by tag or branch, e.g. a new step
    added as `actions/checkout@v6`."""
    text = workflow.read_text()

    assert "uses:" in text, f"{workflow.name} uses no actions"
    assert _unpinned_uses(text) == []


def test_ci_twine_pin_names_the_build_action_release_that_bundles_it() -> None:
    """The lint job's `twine==6.2.0` stands in for the twine the release build runs, which
    is whatever the pinned build-and-inspect release bundles (the v2 line bundles 6.2.0). The
    one-line bug this catches: moving the build action's pin in release.yml (to v3, which
    bundles twine 7) without revisiting the twine pin and the comment that names the release."""
    pins = [
        line.strip()
        for line in _RELEASE_WORKFLOW.read_text().splitlines()
        if "hynek/build-and-inspect-python-package@" in line
    ]
    lint_comment = " ".join(
        line.strip().removeprefix("#").strip()
        for line in _job_block("lint")
        if line.strip().startswith("#")
    )

    assert len(pins) == 1, pins
    tag = pins[0].rsplit("# ", 1)[1]
    assert tag.startswith("v2."), tag
    assert f"hynek/build-and-inspect-python-package {tag}" in lint_comment, lint_comment


def test_ci_lint_job_checks_the_lockfile_between_installing_uv_and_syncing() -> None:
    """Every job syncs with `--frozen`, which installs the lockfile as written even when
    pyproject.toml has moved on, so a dependency change without a lock refresh would test the
    old dependencies and pass. The one-line bug this catches: removing the `uv lock --check`
    step, running it before uv is installed or after the sync, or letting it fail quietly
    (`continue-on-error` or an `if:` on the step, or on the `lint` job itself, which skips or
    excuses every step at once)."""
    block = _job_block("lint")
    job_indent = min(len(line) - len(line.lstrip(" ")) for line in block)
    job_keys = [
        line.split(":", 1)[0].strip()
        for line in block
        if len(line) - len(line.lstrip(" ")) == job_indent and not line.lstrip().startswith("#")
    ]
    assert "continue-on-error" not in job_keys, job_keys
    assert "if" not in job_keys, job_keys
    steps = _job_steps("lint")
    install = _step_indices(steps, "name: Install uv")
    lock = _step_indices(steps, "uv lock --check")
    sync = _step_indices(steps, "run: uv sync")

    assert len(install) == len(lock) == len(sync) == 1, (install, lock, sync)
    assert install[0] < lock[0] < sync[0], (install, lock, sync)
    lock_step = steps[lock[0]]
    assert "run: uv lock --check" in lock_step, lock_step
    keys = [line.removeprefix("- ").split(":", 1)[0].strip() for line in lock_step]
    assert "continue-on-error" not in keys, lock_step
    assert "if" not in keys, lock_step


def test_no_mflux_requirement_has_an_upper_bound() -> None:
    """mflux ships often; the policy is no ceiling: the newest release is tested weekly and
    fixed only when it breaks, so an install never waits on a pin bump here. The one-line bug
    this catches: an upper bound (`<0.26`, `<=0.25`, `==0.24.*`, `~=0.24`) back on any of the
    `mflux` extra, the `showcase` extra or the `test` group."""
    found = _mflux_requirements(_pyproject())

    assert set(found) == {"extra mflux", "extra showcase", "group test"}, sorted(found)
    for place, requirement in found.items():
        operators = {spec.operator for spec in requirement.specifier}
        assert operators <= {">=", ">", "!="}, (place, str(requirement))
        assert requirement.specifier.contains("999.0.0"), (place, str(requirement))


def test_every_pillow_requirement_starts_at_13_with_no_upper_bound() -> None:
    """Pillow 12.3.0 and older are affected by GHSA-5x8w-8jj5-q52h (fixed in 13.0.0), so every
    place that installs Pillow must exclude them and admit every later release. The one-line
    bug this catches: one place left at `>=10.0` (the `image` extra, the `typecheck` or
    `fixtures` group), a floor set past 13.0.0, or a cap such as `<14` back on any of them."""
    pillow = {
        place: Requirement(dep)
        for place, deps in _dependency_places(_pyproject()).items()
        for dep in deps
        if isinstance(dep, str) and Requirement(dep).name.lower() == "pillow"
    }

    assert set(pillow) == {
        "extra image",
        "extra showcase",
        "group test",
        "group typecheck",
        "group fixtures",
    }, sorted(pillow)
    for place, requirement in pillow.items():
        operators = {spec.operator for spec in requirement.specifier}
        assert ">=" in operators, (place, str(requirement))
        assert operators <= {">=", ">", "!="}, (place, str(requirement))
        assert not requirement.specifier.contains("12.3.0"), (place, str(requirement))
        assert not requirement.specifier.contains("12.99.0"), (place, str(requirement))
        assert requirement.specifier.contains("13.0.0"), (place, str(requirement))
        assert requirement.specifier.contains("999.0.0"), (place, str(requirement))


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
    """The lockfile resolves mflux to the newest minor it admits on macOS, so 0.24, the oldest
    minor the release names as supported (`MFLUX_FLOOR_MINOR`), is only exercised by a job
    that installs it over the frozen test sync. The one-line bug
    this catches: deleting the job, or installing the floor before the sync (which would
    put the locked mflux straight back)."""
    job = _job_lines(_MFLUX_FLOOR_JOB)

    sync = job.index("run: uv sync --frozen --group test")
    install = job.index(f'run: uv pip install --no-deps "mflux=={MFLUX_FLOOR_MINOR}.*"')
    assert sync < install


def test_ci_mflux_floor_install_moves_only_mflux() -> None:
    """The floor mflux imports on the locked Pillow 13, so the floor leg installs mflux alone
    and keeps the lock's Pillow. The one-line bug this catches: a Pillow pin (`"Pillow<13"`, the
    old workaround for mflux before 0.24) or any other package left in the floor install."""
    steps = _job_steps(_MFLUX_FLOOR_JOB)
    installs = _step_indices(steps, "uv pip install")

    assert len(installs) == 1, steps
    install = [line for line in steps[installs[0]] if "uv pip install" in line]
    assert install == [f'run: uv pip install --no-deps "mflux=={MFLUX_FLOOR_MINOR}.*"'], install


def test_ci_mflux_floor_job_runs_on_the_lowest_supported_python() -> None:
    """The floor leg pairs the oldest mflux with the oldest Python the release supports, so the
    lowest corner of the support claim runs in CI. The one-line bug this catches: the job's
    `setup-python` left on 3.12 (or any version other than the floor) after the floor moved."""
    versions = [line for line in _job_lines(_MFLUX_FLOOR_JOB) if line.startswith("python-version:")]

    assert versions == [f'python-version: "{SUPPORTED_PYTHONS[0]}"'], versions


_SHELL_OPERATORS = {"&&", "||", ";", "|", "&"}


def _pip_install_arguments(text: str) -> list[str]:
    """Every argument after `pip install` (plain, `uv pip` or `python -m pip`) on any line of
    `text`, up to the end of that shell command (`&&`, `||`, `;`, `|`, `&`), with shell quoting
    removed. Flags are kept; the caller skips them."""
    arguments: list[str] = []
    for line in text.splitlines():
        if "pip install" not in line:
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
        reading = False
        for previous, token in itertools.pairwise(["", *tokens]):
            if token in _SHELL_OPERATORS:
                reading = False
            elif previous == "pip" and token == "install":
                reading = True
            elif reading:
                arguments.append(token)
    return arguments


def _pillow_pins_rejecting_13(text: str) -> list[str]:
    """The `pip install` arguments in `text` that are Pillow requirements whose specifier
    rejects Pillow 13.0.0. Flags and arguments that are not requirements (files, paths) are
    skipped."""
    pins = []
    for argument in _pip_install_arguments(text):
        try:
            requirement = Requirement(argument)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == "pillow" and not requirement.specifier.contains(
            "13.0.0"
        ):
            pins.append(argument)
    return pins


@pytest.mark.parametrize(
    "workflow",
    [_CI_WORKFLOW, _RELEASE_WORKFLOW, _MFLUX_LATEST_WORKFLOW],
    ids=["ci", "release", "mflux-latest"],
)
def test_no_workflow_pins_pillow_below_13(workflow: Path) -> None:
    """Every supported mflux imports on Pillow 13, and Pillow 12.x is affected by
    GHSA-5x8w-8jj5-q52h, so no job installs an older Pillow. The one-line bug this catches:
    the `Pillow<13` workaround left in any workflow's install step, in any spelling
    (`"Pillow < 13"`, `pillow<=12.3`, `pillow==12.*`), or in a check."""
    text = workflow.read_text()

    assert _pillow_pins_rejecting_13(text) == [], workflow.name
    assert "pillow<13" not in text.lower(), workflow.name


@pytest.mark.parametrize(
    "run_line",
    [
        'run: uv pip install --no-deps "mflux==0.24.*" "Pillow < 13"',
        "run: pip install pillow<=12.3",
        "run: uv pip install pillow==12.*",
        "- run: python -m pip install -q 'Pillow<13.0' && uv sync --frozen",
        'run: uv pip install "PILLOW!=13.0.0"',
    ],
)
def test_pillow_pin_reading_flags_every_spelling_that_rejects_13(run_line: str) -> None:
    """The one-line bug this catches: the reader matching only the literal `pillow<13` (so a
    spaced, `<=`, `==12.*` or upper-case pin slips through), reading only `uv pip install` or
    only plain `pip install`, or stopping at the first flag."""
    assert len(_pillow_pins_rejecting_13(run_line)) == 1, run_line


@pytest.mark.parametrize(
    "run_line",
    [
        "run: pip install uv==0.11.2",
        'run: uv pip install --no-deps "mflux==0.24.*"',
        "run: uv pip install --upgrade-package mflux mflux",
        'run: uv pip install "Pillow>=13" pillow',
        "run: pip install -r requirements.txt ./dist/pkg.whl",
        'run: echo "Pillow<12 is not installed here"',
        'run: pip install uv==0.11.2 && echo "pillow==12.*"',
    ],
)
def test_pillow_pin_reading_passes_installs_that_admit_13(run_line: str) -> None:
    """The one-line bug this catches: the reader flagging a Pillow pin that admits 13.0.0, a
    package other than Pillow, a bare `Pillow`, or text outside a `pip install` command (also
    after the command ends at `&&`), or raising on an argument that is not a requirement (a
    file, a path)."""
    assert _pillow_pins_rejecting_13(run_line) == [], run_line


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
    admit. The one-line bug this catches: the comment still naming a minor below the floor
    after the extras moved to `>=0.24`, as `mflux 0.19`, `admit 0.19 and 0.23`, `>=0.19` or
    `>0.23` (the forms `_mflux_minors_named_in` reads), or losing the `(`>=0.24`)` floor."""
    comment = " ".join(
        line.strip().removeprefix("#").strip()
        for line in _job_block(_MFLUX_FLOOR_JOB)
        if line.strip().startswith("#")
    )
    floor = int(MFLUX_FLOOR_MINOR.split(".")[1])
    minors = _mflux_minors_named_in(comment)

    assert f"(`>={MFLUX_FLOOR_MINOR}`)" in comment, comment
    assert minors, comment
    assert all(minor >= floor for minor in minors), comment


def test_floor_comment_reading_skips_decimals_that_are_not_mflux_versions() -> None:
    """The floor-comment test reads every mflux minor the comment names; a duration or ratio
    in the same prose is not one. The one-line bug this catches: a stray `0.N` (e.g. "takes
    0.5 s") read as mflux 0.5, failing the floor check on a correct comment, also when it
    follows a version list's `and`/`or` ("mflux 0.19 and 0.5 s")."""
    comment = (
        "this leg runs the suite on mflux 0.19, the oldest minor the extras admit (`>=0.19`);"
        " the guard takes 0.5 s"
    )

    assert _mflux_minors_named_in(comment) == [19, 19]
    assert _mflux_minors_named_in("mflux 0.19 and 0.5 s") == [19]
    assert _mflux_minors_named_in("mflux 0.19 or 0.5 s, then mflux 0.20") == [19, 20]
    assert _mflux_minors_named_in("mflux 0.19 and 0.5 seconds") == [19]
    assert _mflux_minors_named_in("mflux 0.19 and 0.5 second") == [19]
    assert _mflux_minors_named_in("mflux 0.19, 0.5 sec") == [19]
    assert _mflux_minors_named_in("mflux 0.19 or 0.5 secs, then mflux 0.20") == [19, 20]


def test_floor_comment_reading_still_sees_a_stale_mflux_minor() -> None:
    """Narrowing the reading must not blind it to the claim it exists for. The one-line bug
    this catches: a pattern that no longer matches `mflux 0.N`, `admit 0.N`, a one- or
    two-character operator before `0.N` (`<0.20`, `>=0.19`), or the later minors of a list
    (`0.17, 0.18`, `0.17 and 0.18`, also after a patched item such as `0.17.1, 0.18`), so a
    comment that still names 0.17 or 0.18 passes."""
    assert _mflux_minors_named_in("mflux 0.17.1, 0.18") == [17, 18]
    assert _mflux_minors_named_in("admit 0.19, 0.17.2 and 0.18") == [19, 17, 18]
    assert _mflux_minors_named_in("the pins admit 0.17 and 0.18 so far") == [17, 18]
    assert _mflux_minors_named_in("the pins admit mflux 0.17 (`>=0.17`)") == [17, 17]
    assert _mflux_minors_named_in(
        "this leg runs mflux 0.19 (`>=0.19`); the pins still admit 0.17 and 0.18"
    ) == [19, 19, 17, 18]
    assert _mflux_minors_named_in("the extras resolve mflux 0.17, 0.18 or 0.19") == [17, 18, 19]
    # A word after the list that only starts like a unit is not a duration.
    assert _mflux_minors_named_in("admit 0.17 and 0.18 since v0.8.4") == [17, 18]
    assert _mflux_minors_named_in("admit 0.17 and 0.18 secretly") == [17, 18]
    assert _mflux_minors_named_in("the extras cap mflux at `<0.20`") == [20]
    assert _mflux_minors_named_in("the extras admit `>0.18`") == [18]


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


@pytest.mark.parametrize("pillow_version", ["9.5.0", "10.4.0", "12.99.0"])
def test_ci_mflux_floor_guard_fails_the_job_below_pillow_13(
    monkeypatch: pytest.MonkeyPatch, pillow_version: str
) -> None:
    """The floor leg runs mflux 0.24 on the locked Pillow 13; a Pillow 12 or older that
    slipped in (a lock or resolver change) must fail the job, not pass under it. The one-line
    bug this catches: no Pillow check in the guard, the old `>= 13` rejection left in place, a
    string comparison (`"9.5.0" >= "13"` holds as strings), or `raise SystemExit(0)` instead of
    a failing exit."""
    with pytest.raises(SystemExit) as exit_:
        _run_floor_guard(monkeypatch, _floor_version(0, 1), pillow_version)

    code = exit_.value.code
    assert isinstance(code, str) or (isinstance(code, int) and code != 0), code


@pytest.mark.parametrize("pillow_version", ["13.0.0", "13.1.2", "14.0.0"])
def test_ci_mflux_floor_guard_accepts_every_pillow_major_from_13(
    monkeypatch: pytest.MonkeyPatch, pillow_version: str
) -> None:
    """Any Pillow from 13.0.0 on is what the floor leg should run on. The one-line bug this
    catches: a check off by one major (`> 13`, which rejects 13.0.0), or one that only accepts
    `13.` and rejects Pillow 14."""
    try:
        _run_floor_guard(monkeypatch, _floor_version(0, 1), pillow_version)
    except SystemExit as exit_:
        pytest.fail(f"the guard exited with code {exit_.code!r} on Pillow {pillow_version}")


def test_ci_mflux_floor_guard_prints_the_pillow_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The job log is the record of what the floor leg ran on. The one-line bug this catches:
    the guard's print line losing the Pillow version."""
    _run_floor_guard(monkeypatch, _floor_version(0, 1), "13.0.0")

    assert "Pillow 13.0.0" in capsys.readouterr().out


def test_ci_mflux_floor_job_runs_the_offline_suite_without_a_coverage_gate() -> None:
    """The floor leg must run the same offline selection as the matrix, but a single older
    mflux may legitimately cover different lines, so the 95% gate stays on the matrix. The
    one-line bug this catches: a marker expression that skips the suite (or runs network
    tests) or a copied `--cov-fail-under` that fails the leg for coverage, not behaviour."""
    job = _job_lines(_MFLUX_FLOOR_JOB)
    pytest_runs = [line for line in job if "pytest" in line]

    assert pytest_runs == ['run: uv run --no-sync pytest -m "not benchmark and not network"']
    assert not any("--cov-fail-under" in line for line in job)


@pytest.mark.parametrize(
    ("step", "offline"),
    [
        (["- name: t", "env:", 'HF_HUB_OFFLINE: "1"', "run: uv run pytest"], True),
        (["- name: t", "env:", "HF_HUB_OFFLINE: 1", "run: uv run pytest"], True),
        (["- name: t", "env:", 'HF_HUB_OFFLINE: "0"', "run: uv run pytest"], False),
        (["- name: t", "env:", 'HF_HUB_CACHE: "1"', "run: uv run pytest"], False),
        (["- name: t", "run: HF_HUB_OFFLINE: 1"], False),  # no env key on the step
    ],
)
def test_sets_hf_hub_offline_reads_only_an_env_value_of_1(step: list[str], offline: bool) -> None:
    """The helper the three offline-pytest pins share. The one-line bug this catches: a
    reading that accepts `HF_HUB_OFFLINE: "0"`, another variable, or a step without `env:`."""
    assert _sets_hf_hub_offline(step) is offline


@pytest.mark.parametrize("job_id", ["test", _MFLUX_FLOOR_JOB])
def test_ci_pytest_steps_run_with_the_hub_offline(job_id: str) -> None:
    """The suite is offline by design; `HF_HUB_OFFLINE=1` makes a test that reaches the Hub
    fail instead of downloading. The one-line bug this catches: the `env:` block dropped from
    the pytest step of the `test` matrix or the mflux floor leg."""
    step = _pytest_step(job_id)

    assert _sets_hf_hub_offline(step), step


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


def test_ruff_targets_the_lowest_supported_version() -> None:
    """Ruff's pyupgrade rules and syntax checks read `target-version`; above the floor they
    would accept syntax the floor cannot parse, below it they keep shims the floor no longer
    needs. The one-line bug this catches: `target-version` left on `py310` after the floor
    moved to 3.11, or set to a newer Python than the floor."""
    floor = SUPPORTED_PYTHONS[0].replace(".", "")

    assert _pyproject()["tool"]["ruff"]["target-version"] == f"py{floor}"


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
    Python marker: a marker drops it on some supported Python, a stale `<0.10` cap locks a
    pre-0.14 release, and a floor moved anywhere off 0.14.0 (down to 0.13.5, which a containment check
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


def _latest_steps() -> list[list[str]]:
    return _job_steps(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)


def _latest_step(text: str) -> list[str]:
    """The one step of the latest-mflux job with a line containing `text`."""
    steps = _latest_steps()
    found = _step_indices(steps, text)
    assert len(found) == 1, (text, steps)
    return steps[found[0]]


def test_mflux_latest_workflow_runs_weekly_and_on_demand_only() -> None:
    """mflux has no upper bound here, so a new release reaches users before any pull request
    runs against it; the weekly run is what notices a break. The one-line bug this catches: the
    schedule dropped or made monthly or daily, the manual trigger dropped, or a push or
    pull_request trigger added (which would put an unpinned mflux on every PR)."""
    triggers = _top_level_block("on", _MFLUX_LATEST_WORKFLOW)
    crons = [line for line in triggers if line.startswith("- cron:")]

    assert [line for line in triggers if line.endswith(":")] == [
        "schedule:",
        "workflow_dispatch:",
    ], triggers
    assert len(crons) == 1, triggers
    minute, hour, day_of_month, month, day_of_week = (
        crons[0].removeprefix("- cron:").strip().strip("\"'").split()
    )
    assert minute.isdigit(), crons
    assert hour.isdigit(), crons
    assert (day_of_month, month) == ("*", "*"), crons
    assert day_of_week in {"0", "1", "2", "3", "4", "5", "6"}, crons


def test_mflux_latest_workflow_token_is_read_only() -> None:
    """The job installs and runs whatever the newest mflux release ships, so its token must
    not be able to write. The one-line bug this catches: deleting or widening the top-level
    `permissions:` block, or a job-level `permissions:` that replaces it."""
    assert _top_level_block("permissions", _MFLUX_LATEST_WORKFLOW) == ["contents: read"]
    assert _job_ids(_MFLUX_LATEST_WORKFLOW) == [_MFLUX_LATEST_JOB]
    assert _job_permissions(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW) == []


def test_mflux_latest_checkout_does_not_persist_the_token_and_uv_is_pinned() -> None:
    """Code from the newest mflux runs in this job; the token must not sit in `.git/config`
    for it, and uv must be the version the lockfile was made with. The one-line bug this
    catches: dropping `persist-credentials: false` from the checkout, or unpinning uv."""
    checkout = _latest_step("actions/checkout@")
    installs = [
        line
        for line in _job_lines(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
        if "pip install uv" in line
    ]

    assert "persist-credentials: false" in checkout, checkout
    assert installs == [f"run: pip install uv=={UV_VERSION}"], installs


def test_mflux_latest_job_upgrades_only_mflux_after_the_frozen_sync() -> None:
    """The job starts from the locked test environment and then moves mflux alone to the
    newest release; mlx, Pillow and the rest move only when that mflux's own requirements
    force them. The one-line bug this catches: upgrading before the sync (the sync puts the
    locked mflux back), a bare `--upgrade` (every locked dependency moves, so a red run could
    come from a new Pillow or mlx instead of mflux), `--no-deps` (a release that needs a newer
    Pillow or mlx would not get it), or a second install or sync after the upgrade."""
    steps = _latest_steps()
    sync = _step_indices(steps, "uv sync")
    upgrade = _step_indices(steps, "uv pip install")

    assert len(sync) == len(upgrade) == 1, steps
    assert "run: uv sync --frozen --group test" in steps[sync[0]], steps[sync[0]]
    installs = [line for line in steps[upgrade[0]] if "uv pip install" in line]
    assert installs == ["run: uv pip install --upgrade-package mflux mflux"], installs
    assert sync[0] < upgrade[0], steps


def test_mflux_latest_job_prints_the_versions_it_tests() -> None:
    """The log is the record of which mflux the weekly run tested and with what. The
    one-line bug this catches: dropping the print step or one of the three versions from it,
    or printing before the upgrade (the locked versions, not the tested ones)."""
    steps = _latest_steps()
    printed = _step_indices(steps, 'version("mflux")')
    upgrade = _step_indices(steps, "uv pip install")

    assert len(printed) == 1, steps
    step = " ".join(steps[printed[0]])
    for name in ("mflux", "mlx", "pillow"):
        assert f'"{name}"' in step, (name, step)
    assert upgrade[0] < printed[0], steps


def test_mflux_latest_job_runs_the_offline_suite_without_resyncing() -> None:
    """The suite must run on the upgraded venv, offline. The one-line bug this catches: a
    plain `uv run` (it re-syncs to the lock and tests the locked mflux), a pytest step without
    `HF_HUB_OFFLINE=1` (a test could reach the Hub), or a marker expression that skips the
    suite or runs network or benchmark tests."""
    pytest_step = _latest_step("pytest")
    uv_runs = [
        line for line in _job_lines(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW) if "uv run" in line
    ]

    assert 'run: uv run --no-sync pytest -m "not benchmark and not network"' in pytest_step
    assert _sets_hf_hub_offline(pytest_step), pytest_step
    assert len(uv_runs) >= 2, uv_runs
    assert all("uv run --no-sync " in line for line in uv_runs), uv_runs


def test_mflux_latest_job_cannot_be_switched_off() -> None:
    """A failure here is the signal that a new mflux broke something; it must be loud. The
    one-line bug this catches: `continue-on-error` or an `if:` on the job or a step, or a
    `PYTEST_ADDOPTS` that rewrites the pytest selection."""
    job = _job_lines(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
    keys = [line.removeprefix("- ").split(":", 1)[0].strip() for line in job]

    assert not any("continue-on-error" in line for line in job), job
    assert "if" not in keys, job
    assert not any("PYTEST_ADDOPTS" in line for line in job), job


def test_mflux_latest_job_runs_on_the_test_jobs_runner() -> None:
    """The job is the `test` matrix's suite with the newest mflux; on another runner image a
    failure could come from the platform instead of mflux. The one-line bug this catches:
    moving the `test` job to a new runner and leaving this one on the old one."""
    runs_on = [
        line
        for line in _job_lines(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
        if line.startswith("runs-on:")
    ]

    assert runs_on == [_runs_on("test")], runs_on


def _latest_import_checks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, list[str]]]:
    """(step index, mflux modules requested) for each inline program of the latest-mflux job
    that imports any mflux module, run against a faked mflux 0.25.0 beside Pillow 13.0.0."""
    checks: list[tuple[int, list[str]]] = []
    for step, program, _ in _inline_programs(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW):
        requested = _exec_program(monkeypatch, program, "0.25.0", "13.0.0", "<mflux-latest>")
        if requested:
            checks.append((step, requested))
    return checks


def test_mflux_latest_job_imports_the_mflux_modules_the_suite_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mflux tests `pytest.importorskip` mflux's modules, so a newest mflux that moved or
    broke one would skip every registry walk and leave the weekly job green; only a step that
    imports them turns that red. The one-line bug this catches: no import check, one of the
    four imports dropped, or the check placed before the upgrade (it would import the locked
    mflux) or after pytest."""
    steps = _latest_steps()
    upgrade = _step_indices(steps, "uv pip install")
    pytest_steps = _step_indices(steps, "pytest")
    checks = _latest_import_checks(monkeypatch)

    covering = [step for step, requested in checks if set(_MFLUX_GUARD_IMPORTS) <= set(requested)]
    assert covering, f"no step imports all of {_MFLUX_GUARD_IMPORTS}: {checks}"
    for step, _ in checks:
        assert upgrade[0] < step < pytest_steps[0], (step, steps)


@pytest.mark.parametrize("module", _MFLUX_GUARD_IMPORTS)
def test_mflux_latest_import_check_fails_the_job_when_a_module_does_not_import(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """A module that does not import must fail the step, not be swallowed. The one-line bug
    this catches: wrapping an import in `try`/`except ImportError: pass`, or importing it
    through `importlib.util.find_spec`, which never runs the module."""
    programs = [
        program for _, program, _ in _inline_programs(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
    ]
    raised = [program for program in programs if _raises_import_error(monkeypatch, program, module)]

    assert len(raised) == 1, f"importing {module} failed in {len(raised)} programs, not one"


def _raises_import_error(monkeypatch: pytest.MonkeyPatch, program: str, failing: str) -> bool:
    """Whether the latest-mflux job's `program` raises ImportError when `failing` does not
    import (see `_exec_program`)."""
    try:
        _exec_program(monkeypatch, program, "0.25.0", "13.0.0", "<mflux-latest>", failing)
    except ImportError:
        return True
    return False


def test_mflux_latest_import_check_is_the_whole_step_script() -> None:
    """The import check fails the job only through its exit status, so its step's script must
    be the program alone, run without a re-sync. The one-line bug this catches: `' || true`
    after the closing quote, another command in the same step, or a plain `uv run` that puts
    the locked mflux back before the import."""
    steps = _latest_steps()
    programs = _inline_programs(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
    import_steps = [step for step, program, _ in programs if "import mflux" in program]

    assert len(import_steps) == 1, programs
    step = steps[import_steps[0]]
    assert _GUARD_OPENING in step, step
    assert step[step.index(_GUARD_OPENING) - 1] == "run: |", step
    assert step[-1] == "'", step


def test_mflux_latest_job_times_out_after_an_hour() -> None:
    """Without a job timeout GitHub lets a hung run hold a macOS runner for six hours. The
    one-line bug this catches: dropping the job's `timeout-minutes`, moving it onto one step,
    or a value other than 60."""
    job_keys = [
        line.strip()
        for line in _job_block(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW)
        if len(line) - len(line.lstrip(" ")) == 4 and not line.strip().startswith("#")
    ]

    assert [key for key in job_keys if key.startswith("timeout-minutes:")] == [
        "timeout-minutes: 60"
    ], job_keys


def _comment_text(lines: list[str]) -> str:
    """The `#` comment lines among `lines`, without the `#`, joined by spaces."""
    return " ".join(
        line.strip().removeprefix("#").strip() for line in lines if line.strip().startswith("#")
    )


def test_mflux_latest_job_comment_says_to_check_whether_mlx_moved_first() -> None:
    """Upgrading mflux also moves mlx when the new release needs a newer one, so a red run
    may come from mlx, not mflux. The one-line bug this catches: the triage note dropped from
    the job's comment."""
    comment = _comment_text(_job_block(_MFLUX_LATEST_JOB, _MFLUX_LATEST_WORKFLOW))

    assert "check first whether mlx moved" in comment, comment
