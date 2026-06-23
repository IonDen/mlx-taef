"""Model-kernel package: one self-contained kernel per supported model."""

from types import MappingProxyType

from mlx_taef.errors import UnknownKernelError
from mlx_taef.kernels._types import (
    ArchSpec,
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    UnpackContext,
    WeightSource,
)
from mlx_taef.kernels.flux import TAEF1, TAEF2
from mlx_taef.kernels.qwen import QWEN_IMAGE
from mlx_taef.kernels.sd import TAESD, TAESDXL
from mlx_taef.kernels.zimage import ZIMAGE

# Per-kernel arch builder knobs that are NOT on the shared ArchSpec record.
MIDBLOCK_GN: MappingProxyType[str, bool] = MappingProxyType(
    {
        "taesd": False,
        "taesdxl": False,
        "taef1": False,
        "taef2": True,
        "zimage": False,
        "qwen-image": False,
    }
)

_ALL = (TAESD, TAESDXL, TAEF1, TAEF2, ZIMAGE, QWEN_IMAGE)
KERNELS: MappingProxyType[str, ModelKernel] = MappingProxyType({k.name: k for k in _ALL})


def get_kernel(name: str) -> ModelKernel:
    """Return the kernel registered under `name`, or raise `UnknownKernelError`."""
    try:
        return KERNELS[name]
    except KeyError as e:
        raise UnknownKernelError(f"unknown kernel: {name!r}") from e


__all__ = [
    "KERNELS",
    "MIDBLOCK_GN",
    "ArchSpec",
    "LatentSpec",
    "MfluxBinding",
    "ModelKernel",
    "UnpackContext",
    "WeightSource",
    "get_kernel",
]
