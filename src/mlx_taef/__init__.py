"""mlx-taef: Tiny AutoEncoder family ported to Apple MLX."""

import logging

from mlx_taef.api import TAEF1, TAEF2, TAESD, TAESDXL, Taef, ZImage
from mlx_taef.errors import (
    ConversionError,
    MfluxNotInstalledError,
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
    "MfluxNotInstalledError",
    "Taef",
    "TaefError",
    "UnknownKernelError",
    "ZImage",
    "get_memory_cap_hint",
]

logging.getLogger("mlx_taef").addHandler(logging.NullHandler())
