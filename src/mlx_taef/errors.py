"""Exception classes for mlx-taef.

Hierarchy:
    TaefError                              base for all package-rooted errors
    ├── ConversionError                    HF->MLX conversion dropped/mis-shaped a param
    ├── MfluxNotInstalledError             (+ ImportError) mflux integration dep missing
    ├── MlxTeacacheNotInstalledError       (+ ImportError) showcase teacache dep missing
    ├── UnknownKernelError                 (+ KeyError) name not in the kernel registry
    ├── SchemaVersionError                 raised by the bundled showcase tooling
    └── FixtureLatentMissingError          (+ FileNotFoundError) bundled showcase tooling

SchemaVersionError and FixtureLatentMissingError are raised only by the bundled
showcase tooling (`scripts/run_showcase.py`), not by importable package code.
"""

class TaefError(Exception):
    """Base for all mlx-taef package-rooted errors."""


class SchemaVersionError(TaefError):
    """Raised when loading a JSON report with an unknown schema_version.

    Future schema bumps add adapters; raising rather than silently
    misinterpreting fields keeps the diff workflow honest.
    """


class ConversionError(TaefError):
    """Raised when HF->MLX conversion fails to reproduce the model's parameters.

    Covers two silent-failure paths: an expected parameter that no source key
    produced (would load at random init) and a produced parameter whose shape
    disagrees with the model (would be accepted verbatim). Both yield a
    usable-looking but numerically wrong model, so conversion raises instead.
    """


class MfluxNotInstalledError(TaefError, ImportError):
    """Raised when `mlx_taef.integrations.mflux` is imported but mflux is absent.

    Subclasses TaefError (so `except TaefError` catches it) and ImportError (so
    `except ImportError` keeps working).
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize with a default install-hint message if none is given."""
        if message is None:
            message = (
                "mflux is required for mlx_taef.integrations.mflux. "
                "Install with: pip install 'mlx-taef[mflux]'."
            )
        super().__init__(message)


class MlxTeacacheNotInstalledError(TaefError, ImportError):
    """Raised when a scenario requires `mlx_teacache` but it is not installed.

    Package-rooted error chained from a clear install hint; mirrors
    `MfluxNotInstalledError`.
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize with a default install-hint message if none is given."""
        if message is None:
            message = (
                "mlx-teacache is required for the combined showcase scenario. "
                'Install with `pip install "mlx-taef[showcase]"` or '
                "`uv add mlx-teacache`."
            )
        super().__init__(message)


class FixtureLatentMissingError(TaefError, FileNotFoundError):
    """Raised when a showcase scenario's fixture latent file is missing."""


class UnknownKernelError(TaefError, KeyError):
    """Raised when a kernel name is not in the registry."""
