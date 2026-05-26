"""Exception-class structure tests."""

from __future__ import annotations


def test_taef_error_is_base_exception() -> None:
    from mlx_taef.errors import TaefError

    assert issubclass(TaefError, Exception)


def test_schema_version_error_subclasses_taef_error() -> None:
    from mlx_taef.errors import SchemaVersionError, TaefError

    assert issubclass(SchemaVersionError, TaefError)


def test_mlx_teacache_not_installed_subclasses_taef_error_and_import_error() -> None:
    from mlx_taef.errors import MlxTeacacheNotInstalledError, TaefError

    assert issubclass(MlxTeacacheNotInstalledError, TaefError)
    assert issubclass(MlxTeacacheNotInstalledError, ImportError)


def test_fixture_latent_missing_subclasses_taef_error_and_file_not_found() -> None:
    from mlx_taef.errors import FixtureLatentMissingError, TaefError

    assert issubclass(FixtureLatentMissingError, TaefError)
    assert issubclass(FixtureLatentMissingError, FileNotFoundError)


def test_errors_reexported_from_package_root() -> None:
    import mlx_taef

    assert mlx_taef.TaefError is not None
    assert mlx_taef.SchemaVersionError is not None
    assert mlx_taef.MlxTeacacheNotInstalledError is not None
    assert mlx_taef.FixtureLatentMissingError is not None


def test_mlx_teacache_not_installed_message_mentions_install_path() -> None:
    from mlx_taef.errors import MlxTeacacheNotInstalledError

    e = MlxTeacacheNotInstalledError()
    assert "pip install" in str(e) or "uv add" in str(e)
    assert "mlx-teacache" in str(e)
