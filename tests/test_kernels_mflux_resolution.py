"""Resolving the preview kernel from an mflux model's ModelConfig.

Every case here is offline. The hand-written table carries model names and a sample of the
aliases mflux 0.19.1 registers (read 2026-09-18 from the installed `model_config.py`) so the
resolver is pinned even without mflux; the exhaustive test at the bottom walks the installed
registry itself, so a renamed, added or dropped model in a future mflux turns CI red.
"""

from dataclasses import dataclass, field

import pytest

from mlx_taef import UnsupportedMfluxModelError
from mlx_taef.errors import TaefError
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


# (model_name, a sample of its mflux 0.19.1 aliases) -> expected kernel name, or None when the
# model has no preview kernel here (Lens is the Klein contract but is tracked separately; FIBO,
# ERNIE, Ideogram, Boogu, SeedVR2 have no tiny decoder in this library).
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
    ("krea/Krea-2-Raw", ["krea-2-raw", "krea2-raw"], "krea2"),
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


def test_alias_matching_is_case_insensitive() -> None:
    """Catches: the alias arm comparing raw strings; mflux's own alias lists mix case
    (`flux2-klein-4B`, `klein-9B-kv`), and the alias arm is the one a local checkout with an
    unrecognised name relies on."""
    config = _FakeModelConfig(model_name="/Users/me/models/klein", aliases=["Klein-9B-KV"])
    assert resolve_kernel_from_model_config(config) is KERNELS["taef2"]


def test_krea_2_raw_resolves_by_name_alone() -> None:
    """Catches: the Krea prefix narrowed back to Turbo, so a Krea-2-Raw checkpoint whose
    config carries no aliases (a local path) is rejected although mflux runs it through the
    same VAE and latent creator as Turbo."""
    config = _FakeModelConfig(model_name="krea/Krea-2-Raw")
    assert resolve_kernel_from_model_config(config) is KERNELS["krea2"]


def test_error_class_is_part_of_the_package_root_api() -> None:
    """Catches: `UnsupportedMfluxModelError` importable from the root by accident (imported for
    another reason) but missing from `__all__`, so a star-import or the API docs lose it."""
    import mlx_taef

    assert "UnsupportedMfluxModelError" in mlx_taef.__all__


def test_a_family_prefix_inside_the_name_is_not_a_match() -> None:
    """Catches: `prefix in name` standing in for `name.startswith(prefix)`; a mirror named
    `archive/old-black-forest-labs/FLUX.1-dev-2024` would resolve to taef1 instead of being
    rejected."""
    config = _FakeModelConfig(model_name="archive/old-black-forest-labs/FLUX.1-dev-2024")
    with pytest.raises(UnsupportedMfluxModelError):
        resolve_kernel_from_model_config(config)


def test_base_model_is_consulted_before_model_name_when_they_disagree() -> None:
    """Catches: the prefix arms being ORed across kernels, so a custom checkpoint whose name
    carries one family's prefix while its explicit base model names another dies with a
    "claimed by more than one kernel" error instead of following the base model."""
    config = _FakeModelConfig(
        model_name="black-forest-labs/FLUX.1-dev-klein-merge",
        aliases=["flux2-klein-4b"],
        base_model="black-forest-labs/FLUX.2-klein-4B",
    )
    assert resolve_kernel_from_model_config(config) is KERNELS["taef2"]


def test_a_bare_string_of_aliases_is_not_split_into_characters() -> None:
    """Catches: `aliases="dev"` (a str satisfies Sequence[str]) being iterated into
    `d`, `e`, `v` and a valid FLUX.1 model rejected."""
    assert (
        # a str satisfies Sequence[str] structurally, so no type checker objects to this call
        resolve_kernel_for_mflux(model_name="local/checkout", aliases="dev") is KERNELS["taef1"]
    )


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
    for family in ("FLUX.1", "FLUX.2-klein", "Z-Image", "Qwen-Image", "Krea-2"):
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
            # Registered aliases are lowercase; the resolver folds the model's side.
            assert alias == alias.lower(), (alias, kernel.name)
            assert alias not in seen_aliases, (alias, kernel.name, seen_aliases.get(alias))
            seen_aliases[alias.lower()] = kernel.name
        prefixes.extend(
            (p.lower(), kernel.name) for p in kernel.integration.mflux_model_name_prefixes
        )
    for p1, k1 in prefixes:
        for p2, k2 in prefixes:
            if k1 != k2:
                assert not p1.startswith(p2), (p1, k1, p2, k2)


# Every key of mflux 0.19.1's AVAILABLE_MODELS -> the kernel that previews it, or None. A key
# missing from this map (a model mflux added) fails the exhaustive test with its name, so it has
# to be classified here on purpose rather than claimed or rejected by accident.
_EXPECTED_BY_REGISTRY_KEY: dict[str, str | None] = {
    "dev": "taef1",
    "schnell": "taef1",
    "dev-kontext": "taef1",
    "dev-fill": "taef1",
    "dev-redux": "taef1",
    "dev-depth": "taef1",
    "dev-controlnet-canny": "taef1",
    "schnell-controlnet-canny": "taef1",
    "dev-controlnet-upscaler": "taef1",
    "dev-fill-catvton": "taef1",
    "krea-dev": "taef1",
    "flux2-klein-4b": "taef2",
    "flux2-klein-9b": "taef2",
    "flux2-klein-9b-kv": "taef2",
    "flux2-klein-base-4b": "taef2",
    "flux2-klein-base-9b": "taef2",
    "qwen-image": "qwen-image",
    "qwen-image-edit": "qwen-image",
    "z-image": "zimage",
    "z-image-turbo": "zimage",
    "z-image-turbo-controlnet-union-2.1": "zimage",
    "krea-2": "krea2",
    "krea-2-raw": "krea2",
    "lens-turbo": None,
    "fibo": None,
    "fibo-lite": None,
    "fibo-edit": None,
    "fibo-edit-rmbg": None,
    "ernie-image": None,
    "ernie-image-turbo": None,
    "seedvr2-3b": None,
    "seedvr2-7b": None,
    "ideogram-4-fp8": None,
    "boogu-image-turbo": None,
}


# Alias strings kept from releases before mflux registered its own; not in AVAILABLE_MODELS.
_LEGACY_ALIASES = {"flux1", "flux-dev", "flux-schnell", "flux2"}


def test_every_installed_mflux_registry_entry_is_classified() -> None:
    """Catches: drift between this library and the installed mflux registry in either
    direction: a model mflux added that a broad family prefix now claims (or rejects) without
    anyone deciding, or a model mflux dropped that the map still lists. mflux-built configs
    carry an owner-qualified name, so this walks the prefix arms; the alias arm is covered by
    the two alias tests below."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    available = mflux_config.AVAILABLE_MODELS

    assert set(available) == set(_EXPECTED_BY_REGISTRY_KEY), (
        "classify every new mflux registry key in _EXPECTED_BY_REGISTRY_KEY, and drop removed ones"
    )
    for key, config in available.items():
        expected = _EXPECTED_BY_REGISTRY_KEY[key]
        if expected is None:
            with pytest.raises(UnsupportedMfluxModelError):
                resolve_kernel_from_model_config(config)
        else:
            assert resolve_kernel_from_model_config(config) is KERNELS[expected], key


def test_every_registered_alias_of_a_supported_model_resolves_on_its_own() -> None:
    """Catches: a kernel dropping one of its aliases (`qwen-2512`), which nothing else sees:
    mflux-built configs resolve through the prefix arms, and a config carrying the full alias
    list still matches on the others. Each alias is resolved alone, with a name no prefix
    claims, so the alias arm has to carry it."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")

    for key, config in mflux_config.AVAILABLE_MODELS.items():
        expected = _EXPECTED_BY_REGISTRY_KEY[key]
        for alias in config.aliases:
            local = _FakeModelConfig(model_name=f"/Users/me/models/{key}", aliases=[alias])
            if expected is None:
                with pytest.raises(UnsupportedMfluxModelError):
                    resolve_kernel_from_model_config(local)
            else:
                assert resolve_kernel_from_model_config(local) is KERNELS[expected], alias


def test_every_kernel_alias_is_one_mflux_registers() -> None:
    """Catches: a kernel alias list going stale (mflux renames `zimage-turbo`); the registry
    walk cannot see it because prefixes decide first, and a dead alias is a silent lie about
    what the library supports."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    registered = {a.lower() for c in mflux_config.AVAILABLE_MODELS.values() for a in c.aliases}

    for kernel in KERNELS.values():
        if kernel.integration is None:
            continue
        stale = set(kernel.integration.mflux_models) - registered - _LEGACY_ALIASES
        assert not stale, (kernel.name, sorted(stale))


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
