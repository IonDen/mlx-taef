"""mlx-taef: Tiny AutoEncoder family ported to Apple MLX."""

import logging

from mlx_taef.api import TAEF1, TAEF2, TAESD, TAESDXL, Taef

__all__ = ["TAEF1", "TAEF2", "TAESD", "TAESDXL", "Taef"]

logging.getLogger("mlx_taef").addHandler(logging.NullHandler())
