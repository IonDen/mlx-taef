"""mlx-taef: Tiny AutoEncoder family ported to Apple MLX."""

import logging

from mlx_taef.api import TAEF1, TAEF2, TAESD, TAESDXL, Taef
from mlx_taef.errors import (
    FixtureLatentMissingError,
    MlxTeacacheNotInstalledError,
    SchemaVersionError,
    TaefError,
)

__all__ = [
    "TAEF1",
    "TAEF2",
    "TAESD",
    "TAESDXL",
    "FixtureLatentMissingError",
    "MlxTeacacheNotInstalledError",
    "SchemaVersionError",
    "Taef",
    "TaefError",
]

logging.getLogger("mlx_taef").addHandler(logging.NullHandler())
