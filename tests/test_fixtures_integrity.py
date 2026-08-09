"""Sanity check: committed fixture files match recorded SHA-256."""

import hashlib
import sys

# Version-gated import: no single CI leg runs both arms, so both are excluded
# from coverage rather than reporting one as untested on every interpreter.
if sys.version_info >= (3, 11):  # pragma: no cover
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib
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


def test_every_reference_oracle_is_sha_pinned() -> None:
    # The reference/*.safetensors are the parity oracle: the input latents plus the
    # PyTorch-generated decode/encode ground truth every parity test compares against. Each must
    # have a fixtures.toml entry so a silent oracle replacement (e.g. MLX output saved over the
    # PyTorch reference) is a hard failure here, not a green MLX-vs-MLX comparison (mlx-taef-0005).
    config = tomllib.loads(FIXTURES_TOML.read_text())
    listed = set(config.get("reference", {}))
    on_disk = {p.name for p in (Path(__file__).parent / "reference").glob("*.safetensors")}
    missing = on_disk - listed
    assert not missing, (
        f"reference oracle files missing a fixtures.toml sha entry: {sorted(missing)}"
    )


def test_every_showcase_latent_is_sha_pinned() -> None:
    # Showcase latents back committed SSIM claims (test_zimage_ssim, test_krea2_ssim);
    # a silent replacement must be a hard failure in the DEFAULT suite, not a warning.
    config = tomllib.loads(FIXTURES_TOML.read_text())
    listed = set(config.get("fixtures/showcase_latents", {}))
    on_disk = {
        p.name
        for p in (Path(__file__).parent / "fixtures" / "showcase_latents").glob("*.safetensors")
    }
    missing = on_disk - listed
    assert not missing, f"showcase latents missing a fixtures.toml sha entry: {sorted(missing)}"
