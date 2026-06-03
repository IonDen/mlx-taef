"""Exception-class structure + raise-condition tests.

The structure tests pin the hierarchy (a base-class change reds them). The
raise-condition tests drive a real bad input through the real code path that
raises each error, so a regression where a raise site throws a bare
ImportError / KeyError instead of the package error would be caught — the
declaration-only suite this replaced could not catch that.

`MlxTeacacheNotInstalledError` has no raise site in the package today (a
declared-but-unraised export — see backlog mlx-taef-0009), so it keeps the
structure + default-message companion only; when a real raise site lands, a
condition test belongs here next to the others.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# --- structure: the hierarchy must hold (a base-class change reds these) ---


def test_taef_error_is_base_exception() -> None:
    from mlx_taef.errors import TaefError

    assert issubclass(TaefError, Exception)


def test_schema_version_error_subclasses_taef_error() -> None:
    from mlx_taef.errors import SchemaVersionError, TaefError

    assert issubclass(SchemaVersionError, TaefError)


def test_conversion_error_subclasses_taef_error() -> None:
    from mlx_taef.errors import ConversionError, TaefError

    assert issubclass(ConversionError, TaefError)


def test_mlx_teacache_not_installed_subclasses_taef_error_and_import_error() -> None:
    from mlx_taef.errors import MlxTeacacheNotInstalledError, TaefError

    assert issubclass(MlxTeacacheNotInstalledError, TaefError)
    assert issubclass(MlxTeacacheNotInstalledError, ImportError)


def test_fixture_latent_missing_subclasses_taef_error_and_file_not_found() -> None:
    from mlx_taef.errors import FixtureLatentMissingError, TaefError

    assert issubclass(FixtureLatentMissingError, TaefError)
    assert issubclass(FixtureLatentMissingError, FileNotFoundError)


def test_errors_reexported_from_package_root_by_identity() -> None:
    """Re-export must be the SAME object. The old `is not None` check would
    pass a wrong-object or `True` alias; identity catches a broken re-export.
    Includes ConversionError, which the old check omitted entirely."""
    import mlx_taef
    from mlx_taef import errors

    assert mlx_taef.TaefError is errors.TaefError
    assert mlx_taef.SchemaVersionError is errors.SchemaVersionError
    assert mlx_taef.ConversionError is errors.ConversionError
    assert mlx_taef.MlxTeacacheNotInstalledError is errors.MlxTeacacheNotInstalledError
    assert mlx_taef.FixtureLatentMissingError is errors.FixtureLatentMissingError


# --- raise conditions: a real bad input drives the real raise site ---


def test_verify_conversion_coverage_raises_on_missing_param() -> None:
    """An expected model parameter that no source key produced would load at
    random init; conversion must raise rather than ship a wrong model."""
    from mlx_taef.convert import _verify_conversion_coverage
    from mlx_taef.errors import ConversionError

    with pytest.raises(ConversionError, match=r"missing \d+ expected parameter"):
        _verify_conversion_coverage({}, {"decoder.weight": (4, 4)})


def test_verify_conversion_coverage_raises_on_shape_mismatch() -> None:
    """A produced parameter whose shape disagrees with the model would be
    accepted verbatim; conversion must raise instead."""
    import mlx.core as mx

    from mlx_taef.convert import _verify_conversion_coverage
    from mlx_taef.errors import ConversionError

    converted = {"decoder.weight": mx.zeros((2, 2))}
    with pytest.raises(ConversionError, match="shape mismatch"):
        _verify_conversion_coverage(converted, {"decoder.weight": (4, 4)})


def test_load_report_raises_on_unknown_schema_version(tmp_path: Path) -> None:
    """Loading a report whose schema_version the code doesn't know must raise
    rather than silently misinterpret fields."""
    from scripts.run_showcase import SCHEMA_VERSION, _load_report

    from mlx_taef.errors import SchemaVersionError

    bad = tmp_path / "report.json"
    bad.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "scenarios": {}}))
    with pytest.raises(SchemaVersionError, match="schema_version"):
        _load_report(bad)


def test_check_latent_sha_raises_on_missing_latent(tmp_path: Path) -> None:
    """A showcase scenario pointed at an absent fixture latent must raise the
    package error, not a bare FileNotFoundError from somewhere downstream."""
    from scripts.run_showcase import _check_latent_sha

    from mlx_taef.errors import FixtureLatentMissingError

    with pytest.raises(FixtureLatentMissingError, match="latent missing"):
        _check_latent_sha(tmp_path / "absent_latent.npy")


# --- MlxTeacacheNotInstalledError: no raise site yet (dead export, 0009) ---


def test_mlx_teacache_not_installed_default_message_mentions_install_path() -> None:
    """Companion to a (future) raise-condition test: pins the default install
    hint. This error has no raise site in the package today (mlx-taef-0009), so
    there is no real path to drive a bad input through yet."""
    from mlx_taef.errors import MlxTeacacheNotInstalledError

    e = MlxTeacacheNotInstalledError()
    msg = str(e)
    # both install hints are in the default message; assert both so dropping
    # one is caught (an `or` could not detect a single hint going missing).
    assert "pip install" in msg
    assert "uv add" in msg
    assert "mlx-teacache" in msg
