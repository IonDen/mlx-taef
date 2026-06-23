"""Sanity check: committed fixture files match recorded SHA-256."""

import hashlib
import tomllib
from pathlib import Path

FIXTURES_TOML = Path(__file__).parent / "fixtures.toml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_hashes_match_recorded_sha256() -> None:
    config = tomllib.loads(FIXTURES_TOML.read_text())
    for section, files in config.items():
        for filename, expected_sha in files.items():
            path = Path(__file__).parent / section / filename
            actual = _sha256(path)
            assert actual == expected_sha, (
                f"{path} SHA mismatch: got {actual}, expected {expected_sha}"
            )


def test_every_converted_weight_is_sha_pinned() -> None:
    # Guard against orphaning: every committed converted weight must have a fixtures.toml entry,
    # so removing a sha line can't silently drop a file from integrity checking (the refactor
    # guard no longer covers fixtures.toml).
    config = tomllib.loads(FIXTURES_TOML.read_text())
    listed = set(config.get("converted", {}))
    on_disk = {p.name for p in (Path(__file__).parent / "converted").glob("*.safetensors")}
    missing = on_disk - listed
    assert not missing, f"converted weights missing a fixtures.toml sha entry: {sorted(missing)}"
