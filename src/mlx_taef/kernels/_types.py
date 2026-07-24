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

CONVERTER_VERSION = 1
"""Converted-cache format. Bump whenever source-to-MLX transforms change."""


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
    """SHA-256 for a shared single-file source, verified after download."""
    decoder_sha256: str | None = None
    """Decoder SHA-256 for a two-file source."""
    encoder_sha256: str | None = None
    """Encoder SHA-256 for a two-file source."""

    def __post_init__(self) -> None:
        has_role_sha = self.decoder_sha256 is not None or self.encoder_sha256 is not None
        if self.filename is not None and has_role_sha:
            raise ValueError(
                "single-file sources use sha256; decoder_sha256/encoder_sha256 are only for "
                "two-file sources"
            )
        if self.filename is None and self.sha256 is not None:
            raise ValueError("two-file sources require decoder_sha256 and encoder_sha256")
        if (self.decoder_sha256 is None) != (self.encoder_sha256 is None):
            raise ValueError("decoder_sha256 and encoder_sha256 must be set together")

    def sha256_for(self, role: Role) -> str | None:
        """Return the digest that verifies `role` for this source."""
        if self.filename is not None:
            return self.sha256
        return self.decoder_sha256 if role == "decoder" else self.encoder_sha256

    def cache_key(self, *, role: Role) -> str:
        """Return a stable cache filename stem for (this weight source, role).

        Includes converter version, revision, and the role-selected sha256 so a transform or
        source-pin change never silently reuses stale converted weights.
        """
        if self.filename is not None:
            fname = self.filename
        elif role == "decoder":
            fname = self.decoder_filename or ""
        else:
            fname = self.encoder_filename or ""
        safe_fname = fname.replace("/", "_").replace("..", "_")
        parts = [
            self.repo.replace("/", "_"),
            safe_fname,
            role,
            f"converter-v{CONVERTER_VERSION}",
        ]
        if self.revision is not None:
            parts.append(f"rev-{self.revision[:12]}")
        role_sha256 = self.sha256_for(role)
        if role_sha256 is not None:
            parts.append(f"sha-{role_sha256[:12]}")
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
    midblock_gn: bool = False
    """Whether taesd2d residual blocks include the FLUX.2 GroupNorm branch."""
