from pathlib import Path


def test_ci_uv_sync_commands_are_frozen() -> None:
    workflow = (Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml").read_text()
    sync_commands = [line.strip() for line in workflow.splitlines() if "run: uv sync" in line]

    assert len(sync_commands) == 3
    assert all("--frozen" in command for command in sync_commands)
