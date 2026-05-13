"""HF safetensors -> MLX safetensors conversion.

Zero PyTorch dependency: reads source files with `safetensors.numpy.load_file`
and writes MLX safetensors directly. Runtime users never need torch.
"""

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file as safetensors_load_numpy

from mlx_taef.model import make_decoder
from mlx_taef.variants import TaesdVariantConfig

logger = logging.getLogger(__name__)


def convert_diffusers_to_sequential(
    sd: dict[str, Any],
    *,
    role: str,
) -> dict[str, np.ndarray]:
    """Map Diffusers-VAE keys to upstream Sequential-key format.

    Per the TAEF2 model card, the decoder gets a +1 index shift because the
    Diffusers VAE prepends one layer that the upstream Sequential decoder
    doesn't have. Encoder keys have no offset.

    Args:
        sd: source state dict with Diffusers keys like 'decoder.layers.0.weight'.
        role: 'decoder' (apply +1 offset) or 'encoder' (no offset).

    Returns:
        State dict with upstream-Sequential keys like '0.weight', '1.weight'.
        Keys not matching role-prefix are filtered out.
    """
    out: dict[str, np.ndarray] = {}
    prefix = f"{role}."
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if suffix.startswith("layers."):
            parts = suffix.split(".")
            idx = int(parts[1])
            if role == "decoder":
                idx += 1
            new_key = f"{idx}." + ".".join(parts[2:])
        else:  # pragma: no cover
            new_key = suffix
        out[new_key] = v
    return out


def _load_role_state_dict(  # pragma: no cover
    config: TaesdVariantConfig,
    role: str,
) -> dict[str, np.ndarray]:
    """Download and load weights for (variant, role) into a Sequential-keyed dict."""
    if config.key_format == "diffusers":
        if config.hf_filename is None:
            raise ValueError(f"Diffusers variant {config.name!r} has no hf_filename")
        path = hf_hub_download(repo_id=config.hf_repo, filename=config.hf_filename)
        full_sd = safetensors_load_numpy(path)
        return convert_diffusers_to_sequential(full_sd, role=role)
    if config.key_format == "upstream":
        filename = config.hf_decoder_filename if role == "decoder" else config.hf_encoder_filename
        if filename is None:
            raise ValueError(f"Upstream variant {config.name!r} has no {role} filename")
        path = hf_hub_download(repo_id=config.hf_repo, filename=filename)
        return safetensors_load_numpy(path)
    raise ValueError(f"Unknown key_format: {config.key_format!r}")


def convert_hf_decoder_to_mlx(  # pragma: no cover
    *,
    out_path: Path | str,
    config: TaesdVariantConfig,
) -> None:
    """Download upstream decoder weights, convert to MLX safetensors at `out_path`.

    Handles both upstream-Sequential and Diffusers key formats. Transposes
    Conv2d weights from NCHW to NHWC. Writes the result with MLX-flat keys
    like 'layers.0.weight', 'layers.1.weight', ...

    Args:
        out_path: where to write the MLX safetensors file.
        config: variant configuration.
    """
    sd = _load_role_state_dict(config, role="decoder")
    decoder = make_decoder(config)
    expected = _flatten_module_param_shapes(decoder)
    converted = _build_mlx_state_dict(sd, expected_shapes=expected)
    mx.save_safetensors(str(out_path), converted)


def _sequential_key_to_mlx(src_key: str) -> str:  # pragma: no cover
    """Convert an upstream-Sequential key to an MLX-flat dotted key.

    MLX's `nn.Sequential` stores its children under `.layers`, so every
    integer path segment (after the first top-level layer index) must be
    wrapped as `layers.<N>` rather than a bare `<N>`.

    Examples::

        "1.weight"           -> "layers.1.weight"
        "3.conv.0.weight"    -> "layers.3.conv.layers.0.weight"
        "3.pool.1.bias"      -> "layers.3.pool.layers.1.bias"

    Args:
        src_key: upstream-Sequential key like '3.conv.0.weight'.

    Returns:
        MLX-flat dotted key like 'layers.3.conv.layers.0.weight'.
    """
    parts = src_key.split(".")
    out = ["layers", parts[0]]
    for part in parts[1:]:
        if part.isdigit():
            out.extend(["layers", part])
        else:
            out.append(part)
    return ".".join(out)


def _build_mlx_state_dict(  # pragma: no cover
    sd: dict[str, np.ndarray],
    *,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    """Apply NCHW->NHWC transpose for Conv weights and prefix keys with 'layers.'."""
    converted: dict[str, mx.array] = {}
    for src_key, arr in sd.items():
        dst_key = _sequential_key_to_mlx(src_key)
        if dst_key not in expected_shapes:
            # Skip keys that don't map to the MLX module structure
            # (e.g., extra Diffusers-specific keys we don't need)
            continue
        # Conv2d weight transpose NCHW (out, in, kH, kW) -> NHWC (out, kH, kW, in)
        # Detected when source is 4D and expected MLX shape matches the transposed shape.
        if arr.ndim == 4 and expected_shapes[dst_key] == (
            arr.shape[0],
            arr.shape[2],
            arr.shape[3],
            arr.shape[1],
        ):
            arr = np.transpose(arr, (0, 2, 3, 1)).copy()
        converted[dst_key] = mx.array(arr)
    return converted


def _flatten_module_param_shapes(
    module: Any, prefix: str = ""
) -> dict[str, tuple[int, ...]]:  # pragma: no cover
    """Walk module.parameters() and return a flat dict of dotted-key -> shape."""
    out: dict[str, tuple[int, ...]] = {}

    def _walk(obj: Any, p: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{p}.{k}" if p else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{p}.{i}")
        elif hasattr(obj, "shape"):
            out[p] = tuple(obj.shape)

    _walk(module.parameters(), prefix)
    return out


def convert_hf_encoder_to_mlx(  # pragma: no cover
    *,
    out_path: Path | str,
    config: TaesdVariantConfig,
) -> None:
    """Download upstream encoder weights, convert to MLX safetensors at `out_path`.

    Mirrors `convert_hf_decoder_to_mlx` but introspects via `make_encoder` so
    Conv weights are transposed against the correct shapes.

    Args:
        out_path: where to write the MLX safetensors file.
        config: variant configuration.
    """
    from mlx_taef.model import make_encoder

    sd = _load_role_state_dict(config, role="encoder")
    encoder = make_encoder(config)
    expected = _flatten_module_param_shapes(encoder)
    converted = _build_mlx_state_dict(sd, expected_shapes=expected)
    mx.save_safetensors(str(out_path), converted)


__all__ = [
    "convert_diffusers_to_sequential",
    "convert_hf_decoder_to_mlx",
    "convert_hf_encoder_to_mlx",
]
