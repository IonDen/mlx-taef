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
