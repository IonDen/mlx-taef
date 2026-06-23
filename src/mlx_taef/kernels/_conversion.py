"""Conversion strategies: own the full HF->MLX conversion for a weight source.

Each strategy downloads + key-remaps the source weights, then applies the shared
NCHW->NHWC transpose + coverage-verify (introspecting the built arch module), returning the
arch-shaped MLX state dict. Divergent architectures supply a new strategy here rather than
editing shared convert internals.

The `from mlx_taef.convert import ...` calls are deferred into the methods: importing them at
module load creates a cycle (kernels.__init__ -> flux -> _conversion -> convert -> model ->
variants(shim) -> kernels.KERNELS, which is not yet bound).
"""

import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_taef.errors import ConversionError
from mlx_taef.kernels._types import WeightSource


def _verify_sha256(path: Path, expected: str | None) -> None:
    """Raise ConversionError if the file at `path` doesn't match `expected` sha256 (no-op if None)."""
    if expected is None:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ConversionError(f"sha256 mismatch for {path}: got {digest}, expected {expected}")


class DiffusersRemap:
    """Diffusers single-file source (FLUX.1/FLUX.2/Z-Image). Decoder keys get a +1 offset."""

    def _load_raw(self, source: WeightSource, role: str) -> dict[str, np.ndarray]:
        """Download + key-remap the diffusers single-file source to Sequential keys."""
        from huggingface_hub import hf_hub_download  # pragma: no cover
        from safetensors.numpy import load_file as safetensors_load_numpy  # pragma: no cover

        from mlx_taef.convert import convert_diffusers_to_sequential  # pragma: no cover

        if source.filename is None:  # pragma: no cover
            raise ValueError(f"Diffusers source {source.repo!r} has no filename")
        path = hf_hub_download(repo_id=source.repo, filename=source.filename)  # pragma: no cover
        full_sd = safetensors_load_numpy(path)  # pragma: no cover
        return convert_diffusers_to_sequential(full_sd, role=role)  # pragma: no cover

    def convert(
        self, source: WeightSource, arch_module: object, *, role: str
    ) -> dict[str, mx.array]:
        """Convert the diffusers source for `role` into the arch-shaped MLX state dict."""
        from mlx_taef.convert import _build_mlx_state_dict, _flatten_module_param_shapes

        raw = self._load_raw(source, role)
        expected = _flatten_module_param_shapes(arch_module)
        return _build_mlx_state_dict(raw, expected_shapes=expected)


class UpstreamTwoFile:
    """Upstream two-file source (TAESD/TAESDXL): separate decoder/encoder safetensors."""

    def _load_raw(self, source: WeightSource, role: str) -> dict[str, np.ndarray]:
        """Download the upstream per-role safetensors (already Sequential-keyed)."""
        from huggingface_hub import hf_hub_download  # pragma: no cover
        from safetensors.numpy import load_file as safetensors_load_numpy  # pragma: no cover

        fname = source.decoder_filename if role == "decoder" else source.encoder_filename
        if fname is None:  # pragma: no cover
            raise ValueError(f"Upstream source {source.repo!r} has no {role} filename")
        path = hf_hub_download(repo_id=source.repo, filename=fname)  # pragma: no cover
        return safetensors_load_numpy(path)  # pragma: no cover

    def convert(
        self, source: WeightSource, arch_module: object, *, role: str
    ) -> dict[str, mx.array]:
        """Convert the upstream source for `role` into the arch-shaped MLX state dict."""
        from mlx_taef.convert import _build_mlx_state_dict, _flatten_module_param_shapes

        raw = self._load_raw(source, role)
        expected = _flatten_module_param_shapes(arch_module)
        return _build_mlx_state_dict(raw, expected_shapes=expected)


class TaehvCombined:
    """Combined single-file taew2.1 source: one .safetensors holds both encoder + decoder.

    The MLX `TaehvDecoder`/`TaehvEncoder` are `nn.Sequential` subclasses whose params key as
    `layers.N...` — exactly what `_sequential_key_to_mlx` produces. So conversion is just: keep
    this role's keys, strip the `decoder.`/`encoder.` prefix, cast fp16->fp32 (the canonical
    weights are fp16; parity runs fp32 and MLX would silently upcast fp16 weights anyway), then
    reuse the shared `_build_mlx_state_dict` (NCHW->NHWC conv transpose + strict coverage-verify).
    """

    @staticmethod
    def _select_role(full_sd: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
        """Keep `role`'s tensors, strip the role prefix, cast fp16->fp32 (Sequential-style keys)."""
        prefix = f"{role}."
        return {
            key[len(prefix) :]: arr.astype(np.float32)
            for key, arr in full_sd.items()
            if key.startswith(prefix)
        }

    def _load_raw(self, source: WeightSource, role: str) -> dict[str, np.ndarray]:
        """Download the combined safetensors and select this role's tensors."""
        from huggingface_hub import hf_hub_download  # pragma: no cover
        from safetensors.numpy import load_file as safetensors_load_numpy  # pragma: no cover

        if source.filename is None:  # pragma: no cover
            raise ValueError(f"Combined taehv source {source.repo!r} has no filename")
        path = hf_hub_download(  # pragma: no cover
            repo_id=source.repo, filename=source.filename, revision=source.revision
        )
        _verify_sha256(Path(path), source.sha256)  # pragma: no cover
        return self._select_role(safetensors_load_numpy(path), role)  # pragma: no cover

    def convert(
        self, source: WeightSource, arch_module: object, *, role: str
    ) -> dict[str, mx.array]:
        """Convert the combined taehv source for `role` into the arch-shaped MLX state dict."""
        from mlx_taef.convert import _build_mlx_state_dict, _flatten_module_param_shapes

        raw = self._load_raw(source, role)
        expected = _flatten_module_param_shapes(arch_module)
        return _build_mlx_state_dict(raw, expected_shapes=expected)
