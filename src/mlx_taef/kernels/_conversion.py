"""Conversion strategies: own the full HF->MLX conversion for a weight source.

Each strategy downloads + key-remaps the source weights, then applies the shared
NCHW->NHWC transpose + coverage-verify (introspecting the built arch module), returning the
arch-shaped MLX state dict. Divergent architectures supply a new strategy here rather than
editing shared convert internals.

The `from mlx_taef.convert import ...` calls are deferred into the methods: importing them at
module load creates a cycle (kernels.__init__ -> flux -> _conversion -> convert -> model ->
variants(shim) -> kernels.KERNELS, which is not yet bound).
"""

import mlx.core as mx
import numpy as np

from mlx_taef.kernels._types import WeightSource


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
