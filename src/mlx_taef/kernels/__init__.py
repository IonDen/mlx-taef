"""Model-kernel package: one self-contained kernel per supported model."""

from collections.abc import Iterable, Sequence
from types import MappingProxyType

from mlx_taef.errors import TaefError, UnknownKernelError, UnsupportedMfluxModelError
from mlx_taef.kernels._types import (
    ArchSpec,
    LatentSpec,
    MfluxBinding,
    ModelKernel,
    Role,
    UnpackContext,
    WeightSource,
)
from mlx_taef.kernels.flux import TAEF1, TAEF2
from mlx_taef.kernels.krea2 import KREA2
from mlx_taef.kernels.qwen import QWEN_IMAGE
from mlx_taef.kernels.sd import TAESD, TAESDXL
from mlx_taef.kernels.zimage import ZIMAGE

_ALL = (TAESD, TAESDXL, TAEF1, TAEF2, ZIMAGE, QWEN_IMAGE, KREA2)
KERNELS: MappingProxyType[str, ModelKernel] = MappingProxyType({k.name: k for k in _ALL})
MIDBLOCK_GN: MappingProxyType[str, bool] = MappingProxyType(
    {name: kernel.midblock_gn for name, kernel in KERNELS.items()}
)
"""Compatibility view; architecture construction reads `ModelKernel.midblock_gn`."""


def _claimants_by_prefix(kernels: Sequence[ModelKernel], name: str) -> list[ModelKernel]:
    return [
        k
        for k in kernels
        if k.integration is not None
        and any(name.startswith(p.lower()) for p in k.integration.mflux_model_name_prefixes)
    ]


def _claimants_by_alias(
    kernels: Sequence[ModelKernel], aliases: Sequence[str]
) -> list[ModelKernel]:
    # Registered aliases are lowercase (pinned by the registry tests); fold the model's side.
    wanted = {a.lower() for a in aliases}
    return [
        k
        for k in kernels
        if k.integration is not None
        and any(alias in wanted for alias in k.integration.mflux_models)
    ]


def resolve_kernel_for_mflux(
    *,
    model_name: str | None,
    base_model: str | None = None,
    aliases: Sequence[str] = (),
    kernels: Iterable[ModelKernel] | None = None,
) -> ModelKernel:
    """Return the kernel that previews the mflux model named by these `ModelConfig` fields.

    Three arms are tried in order and the first that claims the model decides: `base_model`
    (the canonical name behind a pinned mirror or a pre-quantized copy) against each binding's
    owner-qualified name prefixes, then `model_name` against the same prefixes, then `aliases`
    against each binding's mflux alias list. Every comparison is case-insensitive. Within an
    arm exactly one kernel may claim the model: more than one raises `TaefError` (a registry
    bug, since two tiny decoders cannot share a family). No arm claiming it raises
    `UnsupportedMfluxModelError`, naming the model and the supported families. `kernels`
    defaults to the shipped registry.
    """
    candidates = tuple(KERNELS.values() if kernels is None else kernels)
    if isinstance(aliases, str):
        aliases = (aliases,)
    described = base_model or model_name or "<unnamed>"
    arms: list[list[ModelKernel]] = [
        _claimants_by_prefix(candidates, n.lower()) for n in (base_model, model_name) if n
    ]
    arms.append(_claimants_by_alias(candidates, aliases))
    for claimants in arms:
        if len(claimants) == 1:
            return claimants[0]
        if claimants:
            raise TaefError(
                f"mflux model {described!r} is claimed by more than one kernel: "
                f"{', '.join(k.name for k in claimants)}"
            )
    families = ", ".join(
        prefix.rstrip("-")
        for k in candidates
        if k.integration is not None
        for prefix in k.integration.mflux_model_name_prefixes
    )
    raise UnsupportedMfluxModelError(
        f"no preview kernel for mflux model {described!r} (aliases {list(aliases)!r}). "
        f"Supported families: {families}. Pass variant= explicitly to choose a decoder for a "
        "model outside these families."
    )


def resolve_kernel_from_model_config(model_config: object) -> ModelKernel:
    """`resolve_kernel_for_mflux` over an mflux `ModelConfig`-shaped object.

    Reads `model_name`, `base_model` and `aliases` by attribute so this module stays
    import-clean of mflux. An object without `model_name` is rejected, not read as unnamed.
    """
    if not hasattr(model_config, "model_name"):
        raise UnsupportedMfluxModelError(
            f"cannot infer the preview variant: {type(model_config).__name__!r} has no "
            "model_name attribute (expected an mflux ModelConfig). Pass variant= explicitly."
        )
    return resolve_kernel_for_mflux(
        model_name=getattr(model_config, "model_name", None),
        base_model=getattr(model_config, "base_model", None),
        aliases=tuple(getattr(model_config, "aliases", ()) or ()),
    )


def get_kernel(name: str) -> ModelKernel:
    """Return the kernel registered under `name`, or raise `UnknownKernelError`."""
    try:
        return KERNELS[name]
    except KeyError as e:
        raise UnknownKernelError(f"unknown kernel: {name!r}") from e


__all__ = [
    "KERNELS",
    "MIDBLOCK_GN",
    "ArchSpec",
    "LatentSpec",
    "MfluxBinding",
    "ModelKernel",
    "Role",
    "UnpackContext",
    "WeightSource",
    "get_kernel",
    "resolve_kernel_for_mflux",
    "resolve_kernel_from_model_config",
]
