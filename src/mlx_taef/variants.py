"""Variant configurations for the TAESD-family of tiny autoencoders.

Field reference:
- `key_format`: "upstream" means the HF safetensors uses upstream Sequential keys
  like "0.weight", "1.weight"; "diffusers" means the HF safetensors uses Diffusers
  VAE keys like "encoder.layers.0.weight" and requires a +1 decoder index offset.
- `arch_variant`: None | "flux_2" | "f32". "flux_2" enables midblock_gn pool
  branches in three blocks of encoder and decoder. "f32" is reserved for
  TAESANA (future).
- `latent_magnitude`, `latent_shift`: from upstream TAESD.scale_latents /
  unscale_latents.
- `hf_filename` (Diffusers single-file format) vs `hf_decoder_filename` /
  `hf_encoder_filename` (upstream two-file format) — depends on the actual
  repo layout for each variant.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaesdVariantConfig:
    """Configuration for a single TAESD-family variant.

    Attributes:
        name: short identifier like "taef2".
        latent_channels: latent channel count (4 for SD1/SDXL, 16 for FLUX.1, 32 for FLUX.2 Klein).
        arch_variant: None for the standard arch, "flux_2" for TAEF2's midblock_gn, "f32" for TAESANA.
        key_format: "upstream" (Sequential keys) or "diffusers" (requires remap).
        hf_repo: e.g. "madebyollin/taef2".
        hf_filename: filename within hf_repo when key_format == "diffusers" (single file).
        hf_decoder_filename: decoder filename when key_format == "upstream".
        hf_encoder_filename: encoder filename when key_format == "upstream".
        latent_magnitude: scale factor for scale_latents/unscale_latents.
        latent_shift: shift factor for scale_latents/unscale_latents.
    """

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


TAESD_CONFIG = TaesdVariantConfig(
    name="taesd",
    latent_channels=4,
    arch_variant=None,
    key_format="upstream",
    hf_repo="madebyollin/taesd",
    hf_filename=None,
    hf_decoder_filename="taesd_decoder.safetensors",
    hf_encoder_filename="taesd_encoder.safetensors",
)

TAESDXL_CONFIG = TaesdVariantConfig(
    name="taesdxl",
    latent_channels=4,
    arch_variant=None,
    key_format="upstream",
    hf_repo="madebyollin/taesdxl",
    hf_filename=None,
    hf_decoder_filename="taesdxl_decoder.safetensors",
    hf_encoder_filename="taesdxl_encoder.safetensors",
)

TAEF1_CONFIG = TaesdVariantConfig(
    name="taef1",
    latent_channels=16,
    arch_variant=None,
    key_format="diffusers",
    hf_repo="madebyollin/taef1",
    hf_filename="diffusion_pytorch_model.safetensors",
    hf_decoder_filename=None,
    hf_encoder_filename=None,
    memory_cap_hint_gb=1,
)

TAEF2_CONFIG = TaesdVariantConfig(
    name="taef2",
    latent_channels=32,
    arch_variant="flux_2",
    key_format="diffusers",
    hf_repo="madebyollin/taef2",
    hf_filename="taef2.safetensors",
    hf_decoder_filename=None,
    hf_encoder_filename=None,
    memory_cap_hint_gb=2,
)

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
        cfg = VARIANTS[variant]
    except KeyError as e:
        raise KeyError(f"unknown variant: {variant!r}") from e
    return cfg.memory_cap_hint_gb


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
