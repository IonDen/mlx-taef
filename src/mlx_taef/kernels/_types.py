"""Composable model-kernel types.

A `ModelKernel` is a frozen record of small strategy objects: an `ArchSpec` (names a shared
arch builder), a `ConversionStrategy` (owns the full HF->MLX conversion), a `LatentSpec`
(latent metadata), a `WeightSource` (where upstream weights live + how they cache), and an
optional `MfluxBinding` (which mflux models it previews + how to unpack their in-loop latent).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx


@dataclass(frozen=True, slots=True, kw_only=True)
class LatentSpec:
    """Latent metadata (scalar scale/shift; FLUX/Z-Image form).

    A per-channel-stats variant is added in Phase 3 (Qwen/Wan) when that contract is known.
    """

    channels: int
    magnitude: float = 3.0
    shift: float = 0.5
    downsample: int = 8


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchSpec:
    """Names a shared architecture builder.

    Channel count flows from `LatentSpec` into the builder at construction; builder-specific
    knobs live builder-local, not here.
    """

    name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightSource:
    """Upstream weights location + cache identity.

    `cache_key` derives from (repo, filename, role) — NOT the kernel name — so kernels that
    share source weights (e.g. zimage -> taef1) share one converted-cache entry, while a
    model's decoder and encoder never collide (single-file sources reuse one filename).
    """

    repo: str
    filename: str | None = None
    decoder_filename: str | None = None
    encoder_filename: str | None = None
    revision: str | None = None
    """Optional HF commit revision (full SHA) to pin the download to a fixed checkpoint."""
    sha256: str | None = None
    """Optional sha256 of the source file; verified after download when set (supply-chain pin)."""

    def cache_key(self, *, role: str) -> str:
        """Return a stable cache filename stem for (this weight source, role)."""
        if self.filename is not None:
            fname = self.filename
        elif role == "decoder":
            fname = self.decoder_filename or ""
        else:
            fname = self.encoder_filename or ""
        safe_fname = fname.replace("/", "_").replace("..", "_")
        return f"{self.repo.replace('/', '_')}__{safe_fname}__{role}"


@dataclass(frozen=True, slots=True, kw_only=True)
class UnpackContext:
    """Everything an unpack callable may need, assembled once by the callback.

    The callback itself does zero model-name branching.
    """

    latent_height: int
    latent_width: int
    bn_mean: mx.array | None = None
    bn_var: mx.array | None = None
    bn_eps: float = 1e-4


@dataclass(frozen=True, slots=True, kw_only=True)
class MfluxBinding:
    """Binds a kernel to the mflux model(s) it previews and how to unpack their latent."""

    mflux_models: tuple[str, ...]
    unpack: Callable[[mx.array, UnpackContext], mx.array]
    packed_latent_downscale: int | None = 16
    """Image-pixels per packed in-loop latent cell, for auto-resolution. FLUX in-loop latents
    are 2x2-PACKED on top of the 8x VAE, so this is 8*2 = 16 (FLUX.1, FLUX.2) and
    latent_height = image_height // 16.

    This is NOT the VAE spatial scale (that is `LatentSpec.downsample`); it is the packed-latent
    ratio. `None` means the in-loop latent is NOT packed and already carries its own spatial dims
    (Z-Image: `(16, 1, h, w)`) — the callback then skips config-derived resolution entirely. If a
    future unpack for such a model ever consumes `ctx` dims, set this to its image/latent ratio
    (8 for Z-Image) instead of `None`."""


class ConversionStrategy(Protocol):
    """Owns the full HF->MLX conversion.

    Handles download + key-remap + NCHW->NHWC transpose + coverage-verify,
    returning the arch-shaped MLX state dict.
    """

    def convert(
        self, source: WeightSource, arch_module: object, *, role: str
    ) -> dict[str, mx.array]:
        """Convert the source weights for `role` into the arch-shaped MLX state dict."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelKernel:
    """One self-contained model: arch + conversion + latent + weights + mflux binding."""

    name: str
    arch: ArchSpec
    conversion: ConversionStrategy
    latent: LatentSpec
    source: WeightSource
    integration: MfluxBinding | None = None
    memory_cap_hint_gb: int | None = None
