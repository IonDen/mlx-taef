"""Resolving the preview kernel from an mflux model's ModelConfig.

Every case here is offline: the fake configs carry the exact strings mflux 0.19.1's registry
pins (verified 2026-09-18 against the installed `model_config.py`), and the one test that uses
mflux's real `ModelConfig` objects imports them without loading weights.
"""

from dataclasses import dataclass, field

import pytest

from mlx_taef.errors import TaefError, UnsupportedMfluxModelError
from mlx_taef.kernels import (
    KERNELS,
    resolve_kernel_for_mflux,
    resolve_kernel_from_model_config,
)


@dataclass
class _FakeModelConfig:
    """The three attributes mflux's ModelConfig exposes that the resolver reads."""

    model_name: str
    aliases: list[str] = field(default_factory=list)
    base_model: str | None = None


# (model_name, aliases) exactly as mflux 0.19.1 registers them -> expected kernel name, or None
# when the model has no preview kernel here (Lens is the Klein contract but is tracked separately;
# FIBO, ERNIE, Ideogram, Boogu, SeedVR2 have no tiny decoder in this library).
_REGISTRY_CASES: list[tuple[str, list[str], str | None]] = [
    ("black-forest-labs/FLUX.1-dev", ["dev"], "taef1"),
    ("black-forest-labs/FLUX.1-schnell", ["schnell"], "taef1"),
    ("black-forest-labs/FLUX.1-Kontext-dev", ["dev-kontext"], "taef1"),
    ("black-forest-labs/FLUX.1-Fill-dev", ["dev-fill"], "taef1"),
    ("black-forest-labs/FLUX.1-Redux-dev", ["dev-redux"], "taef1"),
    ("black-forest-labs/FLUX.1-Depth-dev", ["dev-depth"], "taef1"),
    ("black-forest-labs/FLUX.1-Krea-dev", ["krea-dev", "dev-krea"], "taef1"),
    (
        "black-forest-labs/FLUX.2-klein-4B",
        ["flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B"],
        "taef2",
    ),
    ("black-forest-labs/FLUX.2-klein-9B", ["flux2-klein-9b", "klein-9b"], "taef2"),
    ("black-forest-labs/FLUX.2-klein-9b-kv", ["flux2-klein-9b-kv", "klein-9b-kv"], "taef2"),
    ("black-forest-labs/FLUX.2-klein-base-4B", ["flux2-klein-base-4b", "flux2-base-4b"], "taef2"),
    ("black-forest-labs/FLUX.2-klein-base-9B", ["flux2-klein-base-9b", "klein-base-9B"], "taef2"),
    ("Qwen/Qwen-Image-2512", ["qwen-image", "qwen", "qwen-image-2512", "qwen-2512"], "qwen-image"),
    ("Qwen/Qwen-Image-Edit-2509", ["qwen-image-edit", "qwen-edit", "qwen-edit-2511"], "qwen-image"),
    ("Tongyi-MAI/Z-Image", ["z-image", "zimage"], "zimage"),
    ("Tongyi-MAI/Z-Image-Turbo", ["z-image-turbo", "zimage-turbo"], "zimage"),
    ("Tongyi-MAI/Z-Image-Turbo", ["z-image-turbo-controlnet", "z-image-controlnet"], "zimage"),
    ("krea/Krea-2-Turbo", ["krea-2", "krea2"], "krea2"),
    ("krea/Krea-2-Raw", ["krea-2-raw", "krea2-raw"], None),
    ("Comfy-Org/Lens", ["lens-turbo", "lens"], None),
    ("briaai/FIBO", ["fibo"], None),
    ("baidu/ERNIE-Image", ["ernie-image"], None),
    ("ideogram-ai/ideogram-4-fp8", ["ideogram-4", "ideogram"], None),
    ("Boogu/Boogu-Image-0.1-Turbo", ["boogu-image", "boogu"], None),
    ("numz/SeedVR2_comfyUI", ["seedvr2-3b", "seedvr2"], None),
]


@pytest.mark.parametrize(("model_name", "aliases", "expected"), _REGISTRY_CASES)
def test_every_mflux_registry_entry_resolves_to_its_kernel_or_is_rejected(
    model_name: str, aliases: list[str], expected: str | None
) -> None:
    """Catches: a family prefix or alias list that maps a registry model to the wrong tiny
    decoder (wrong latent channels at the first preview step), or that accepts a model with
    no preview kernel here and hands it taef2 silently."""
    config = _FakeModelConfig(model_name=model_name, aliases=aliases)
    if expected is None:
        with pytest.raises(UnsupportedMfluxModelError, match=model_name.replace(".", r"\.")):
            resolve_kernel_from_model_config(config)
    else:
        assert resolve_kernel_from_model_config(config) is KERNELS[expected]


def test_base_model_wins_over_a_mirror_model_name() -> None:
    """Catches: a pre-quantized mirror (`someone/FLUX.2-klein-4B-mlx-4bit`) whose canonical
    `base_model` is set but whose `model_name` matches nothing, rejected instead of resolved."""
    config = _FakeModelConfig(
        model_name="someone/klein-4B-mlx-4bit",
        aliases=[],
        base_model="black-forest-labs/FLUX.2-klein-4B",
    )
    assert resolve_kernel_from_model_config(config) is KERNELS["taef2"]


def test_aliases_resolve_a_local_path_model_name() -> None:
    """Catches: a user-pinned local checkout (`model_name` is a filesystem path) with a known
    alias list being rejected because only the name was consulted."""
    config = _FakeModelConfig(model_name="/Users/me/models/zimage-turbo", aliases=["z-image-turbo"])
    assert resolve_kernel_from_model_config(config) is KERNELS["zimage"]


def test_matching_is_case_insensitive_on_the_name() -> None:
    """Catches: `Tongyi-MAI/z-image-turbo` (a lowercase mirror) failing a case-sensitive prefix."""
    config = _FakeModelConfig(model_name="tongyi-mai/z-image-turbo")
    assert resolve_kernel_from_model_config(config) is KERNELS["zimage"]


def test_unknown_model_error_names_the_model_and_every_supported_family() -> None:
    """Catches: an error that says "unsupported" without telling the user what they passed or
    which families would have worked."""
    config = _FakeModelConfig(model_name="acme/NewModel-1", aliases=["newmodel"])
    with pytest.raises(UnsupportedMfluxModelError) as excinfo:
        resolve_kernel_from_model_config(config)
    message = str(excinfo.value)
    assert "acme/NewModel-1" in message
    for family in ("FLUX.1", "FLUX.2-klein", "Z-Image", "Qwen-Image", "Krea-2-Turbo"):
        assert family in message
    assert "variant=" in message


def test_config_without_the_expected_attributes_is_rejected_not_guessed() -> None:
    """Catches: an object with no `model_name` (an mflux fork, or the wrong argument) being
    read as an empty name and falling through to a default."""
    with pytest.raises(UnsupportedMfluxModelError, match="model_name"):
        resolve_kernel_from_model_config(object())


def test_two_kernels_claiming_one_model_is_a_registry_error() -> None:
    """Catches: two bindings whose prefixes overlap (a future `FLUX.2-klein-video` kernel next
    to `FLUX.2-klein`) silently resolving to whichever was registered first."""
    from dataclasses import replace

    a = KERNELS["taef1"]
    b = replace(
        KERNELS["taef2"],
        name="taef2-shadow",
        integration=replace(
            KERNELS["taef2"].integration, mflux_model_name_prefixes=("black-forest-labs/FLUX.1-",)
        ),
    )
    with pytest.raises(TaefError, match=r"taef1.*taef2-shadow|taef2-shadow.*taef1"):
        resolve_kernel_for_mflux(
            model_name="black-forest-labs/FLUX.1-dev", base_model=None, aliases=(), kernels=(a, b)
        )


def test_registry_prefixes_and_aliases_are_disjoint_across_kernels() -> None:
    """Catches: the same alias or an overlapping prefix registered on two shipped kernels."""
    seen_aliases: dict[str, str] = {}
    prefixes: list[tuple[str, str]] = []
    for kernel in KERNELS.values():
        if kernel.integration is None:
            continue
        for alias in kernel.integration.mflux_models:
            assert alias.lower() not in seen_aliases, (alias, kernel.name, seen_aliases.get(alias))
            seen_aliases[alias.lower()] = kernel.name
        prefixes.extend(
            (p.lower(), kernel.name) for p in kernel.integration.mflux_model_name_prefixes
        )
    for p1, k1 in prefixes:
        for p2, k2 in prefixes:
            if k1 != k2:
                assert not p1.startswith(p2), (p1, k1, p2, k2)


def test_real_mflux_model_configs_resolve() -> None:
    """Catches: drift between the strings this module pins and what the installed mflux
    actually registers (a renamed model_name or alias in a new mflux)."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    model_config_cls = mflux_config.ModelConfig

    expected = {
        "dev": "taef1",
        "schnell": "taef1",
        "flux2-klein-base-4b": "taef2",
        "flux2-klein-9b": "taef2",
        "z-image-turbo": "zimage",
        "qwen-image": "qwen-image",
        "krea-2": "krea2",
    }
    for alias, kernel_name in expected.items():
        config = model_config_cls.from_name(model_name=alias, base_model=None)
        assert resolve_kernel_from_model_config(config) is KERNELS[kernel_name], alias
    with pytest.raises(UnsupportedMfluxModelError):
        resolve_kernel_from_model_config(
            model_config_cls.from_name(model_name="lens", base_model=None)
        )
