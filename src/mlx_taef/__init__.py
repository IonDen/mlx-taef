"""mlx-taef: Tiny AutoEncoder family ported to Apple MLX."""

import logging

from mlx_taef.api import TAEF1, TAEF2, TAESD, TAESDXL, Taef
from mlx_taef.errors import (
    ConversionError,
    FixtureLatentMissingError,
    MfluxNotInstalledError,
    MlxTeacacheNotInstalledError,
    SchemaVersionError,
    TaefError,
    UnknownKernelError,
)
from mlx_taef.variants import get_memory_cap_hint

__all__ = [
    "TAEF1",
    "TAEF2",
    "TAESD",
    "TAESDXL",
    "ConversionError",
    "FixtureLatentMissingError",
    "MfluxNotInstalledError",
    "MlxTeacacheNotInstalledError",
    "SchemaVersionError",
    "Taef",
    "TaefError",
    "UnknownKernelError",
    "get_memory_cap_hint",
]

logging.getLogger("mlx_taef").addHandler(logging.NullHandler())
