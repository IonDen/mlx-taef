"""Composable model-kernel types.

A `ModelKernel` is a frozen record of small strategy objects: an `ArchSpec` (names a shared
arch builder), a `ConversionStrategy` (owns the full HF->MLX conversion), a `LatentSpec`
(latent metadata), a `WeightSource` (where upstream weights live + how they cache), and an
optional `MfluxBinding` (which mflux models it previews + how to unpack their in-loop latent).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import mlx.core as mx

Role = Literal["decoder", "encoder"]
"""The two weight roles a kernel converts/caches independently."""


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

    def __post_init__(self) -> None:
        # A single sha256 cannot verify two distinct role files. Reject a sha256 pin on a
        # two-file (decoder/encoder) source until per-role digests exist — otherwise one role
        # would silently verify the wrong file's digest. Only single-file sources are pinned
        # today; this guards a future two-file pin.
        if self.sha256 is not None and self.filename is None:
            raise ValueError(
                "sha256 pinning is only supported for single-file sources (filename=...). "
                "This two-file source (decoder_filename/encoder_filename) would verify both "
                "roles against one digest; add per-role digests before pinning sha256."
            )

    def cache_key(self, *, role: Role) -> str:
        """Return a stable cache filename stem for (this weight source, role).

        Includes revision + sha256 (when pinned) so bumping a source's pinned revision or
        sha256 changes the key — an existing cache is never silently reused for a new pin.
        Unpinned sources (taef1/taef2/taesd/taesdxl) keep their pre-0.6.2 key, so they do
        NOT re-convert on upgrade.
        """
        if self.filename is not None:
            fname = self.filename
        elif role == "decoder":
            fname = self.decoder_filename or ""
        else:
            fname = self.encoder_filename or ""
        safe_fname = fname.replace("/", "_").replace("..", "_")
        parts = [self.repo.replace("/", "_"), safe_fname, role]
        if self.revision is not None:
            parts.append(f"rev-{self.revision[:12]}")
        if self.sha256 is not None:
            parts.append(f"sha-{self.sha256[:12]}")
        return "__".join(parts)


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
        self, source: WeightSource, arch_module: object, *, role: Role
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
