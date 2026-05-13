"""HF Hub auto-download + cache. Zero PyTorch dependency at runtime."""

import logging
from pathlib import Path

from mlx_taef.convert import convert_hf_decoder_to_mlx, convert_hf_encoder_to_mlx
from mlx_taef.variants import TaesdVariantConfig

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "mlx-taef"


def get_or_convert(config: TaesdVariantConfig, *, role: str = "decoder") -> Path:
    """Return the local path to converted MLX weights for (variant, role).

    On cache miss, triggers the full conversion pipeline (HF download + key
    remap + NHWC transpose + safetensors write). Subsequent calls return the
    cached path without any network access.

    Args:
        config: variant configuration (selects HF repo + filename).
        role: 'decoder' (default) or 'encoder'.

    Returns:
        Local filesystem path to the MLX safetensors file.
    """
    cache_dir = CACHE_ROOT / config.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{config.name}_{role}.safetensors"
    if out_path.exists():
        logger.debug("Using cached weights at %s", out_path)
        return out_path

    logger.info("Downloading + converting %s %s weights from %s", config.name, role, config.hf_repo)
    if role == "decoder":
        convert_hf_decoder_to_mlx(out_path=out_path, config=config)
    elif role == "encoder":
        convert_hf_encoder_to_mlx(out_path=out_path, config=config)
    else:
        raise ValueError(f"role must be 'decoder' or 'encoder', got {role!r}")
    return out_path


__all__ = ["CACHE_ROOT", "get_or_convert"]
