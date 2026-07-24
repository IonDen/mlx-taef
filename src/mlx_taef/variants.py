"""Back-compat shim. The source of truth moved to `mlx_taef.kernels`.

Reconstructs the legacy `TaesdVariantConfig` view + `*_CONFIG` constants, `ALL_VARIANTS`,
`VARIANTS`, and `get_memory_cap_hint` from the kernel registry so existing imports keep
working. New code should import from `mlx_taef.kernels`.
"""

import logging
from dataclasses import dataclass

from mlx_taef.errors import UnknownKernelError
from mlx_taef.kernels import KERNELS, ModelKernel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaesdVariantConfig:
    """Legacy variant config view (derived from a ModelKernel)."""

    name: str
    latent_channels: int
    arch_variant: str | None
    key_format: str
    hf_repo: str
    hf_filename: str | None
    hf_decoder_filename: str | None
    hf_encoder_filename: str | None
    latent_magnitude: float = 3.0
    latent_shift: float = 0.5
    memory_cap_hint_gb: int | None = None

    @property
    def use_midblock_gn(self) -> bool:
        """Whether the variant's Block layers use the midblock GroupNorm pool branch."""
        return self.arch_variant == "flux_2"


def _from_kernel(k: ModelKernel) -> TaesdVariantConfig:
    is_diffusers = k.source.filename is not None
    return TaesdVariantConfig(
        name=k.name,
        latent_channels=k.latent.channels,
        arch_variant="flux_2" if k.midblock_gn else None,
        key_format="diffusers" if is_diffusers else "upstream",
        hf_repo=k.source.repo,
        hf_filename=k.source.filename,
        hf_decoder_filename=k.source.decoder_filename,
        hf_encoder_filename=k.source.encoder_filename,
        latent_magnitude=k.latent.magnitude,
        latent_shift=k.latent.shift,
        memory_cap_hint_gb=k.memory_cap_hint_gb,
    )


TAESD_CONFIG = _from_kernel(KERNELS["taesd"])
TAESDXL_CONFIG = _from_kernel(KERNELS["taesdxl"])
TAEF1_CONFIG = _from_kernel(KERNELS["taef1"])
TAEF2_CONFIG = _from_kernel(KERNELS["taef2"])

ALL_VARIANTS: tuple[TaesdVariantConfig, ...] = (
    TAESD_CONFIG,
    TAESDXL_CONFIG,
    TAEF1_CONFIG,
    TAEF2_CONFIG,
)
VARIANTS: dict[str, TaesdVariantConfig] = {v.name: v for v in ALL_VARIANTS}


def get_memory_cap_hint(variant: str) -> int | None:
    """Return the per-variant `memory_cap_hint_gb` (GB) or None.

    Raises:
        KeyError: if `variant` is not a known variant name.
    """
    try:
        kernel = KERNELS[variant]
    except KeyError as e:
        raise UnknownKernelError(f"unknown variant: {variant!r}") from e
    return kernel.memory_cap_hint_gb


__all__ = [
    "ALL_VARIANTS",
    "TAEF1_CONFIG",
    "TAEF2_CONFIG",
    "TAESDXL_CONFIG",
    "TAESD_CONFIG",
    "VARIANTS",
    "TaesdVariantConfig",
    "get_memory_cap_hint",
]
