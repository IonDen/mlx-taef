"""HF Hub auto-download + cache. Zero PyTorch dependency at runtime."""

import logging
from pathlib import Path

import mlx.core as mx

from mlx_taef.kernels import MIDBLOCK_GN, ModelKernel
from mlx_taef.kernels._arch import build_arch

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "mlx-taef"


def get_or_convert(kernel: ModelKernel, *, role: str = "decoder") -> Path:
    """Return the local path to converted MLX weights for (kernel, role).

    Cache is keyed on the weight SOURCE identity (repo + filename + role), not the kernel
    name, so kernels that share upstream weights (e.g. zimage -> taef1) share one converted
    file while a model's decoder/encoder never collide.
    """
    if role not in ("decoder", "encoder"):
        raise ValueError(f"role must be 'decoder' or 'encoder', got {role!r}")
    cache_dir = CACHE_ROOT / "converted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{kernel.source.cache_key(role=role)}.mlx.safetensors"
    if out_path.exists():
        logger.debug("Using cached weights at %s", out_path)
        return out_path

    logger.info("Downloading + converting %s %s from %s", kernel.name, role, kernel.source.repo)
    ch = kernel.latent.channels
    mbgn = MIDBLOCK_GN.get(kernel.name, False)
    arch_module = build_arch(kernel.arch.name, role=role, latent_channels=ch, midblock_gn=mbgn)
    converted = kernel.conversion.convert(kernel.source, arch_module, role=role)
    mx.save_safetensors(str(out_path), converted)
    return out_path


__all__ = ["CACHE_ROOT", "get_or_convert"]
