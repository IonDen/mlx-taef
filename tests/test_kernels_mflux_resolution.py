"""Resolving the preview kernel from an mflux model's ModelConfig.

Every case here is offline. The hand-written table carries model names and a sample of the
aliases mflux 0.19.1 registers (read 2026-09-18 from the installed `model_config.py`) so the
resolver is pinned even without mflux; the exhaustive tests at the bottom walk the installed
registry itself. A routing change in a future mflux turns CI red: a model this file has not
classified that some kernel now claims, a classified model whose kernel changes, or a model mflux
dropped. A new model the resolver refuses is listed in an `UnclassifiedMfluxModelsWarning` (in the
pytest warnings summary) without failing, until someone classifies it here.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

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
    ("Qwen/Qwen-Image-2.1", ["qwen-image-2.1", "qwen-2.1", "qwen-image-21"], None),
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


# Registry keys mflux 0.21 to 0.23 added, as mflux 0.23.0's `model_config.py` builds them
# (model_name, aliases; base_model is None for all), read 2026-10-23 from the tagged source
# -> expected kernel name, or None when no tiny decoder here previews the model. Alias lists are
# copied whole, in mflux's order (the key-named alias first); a guard test below compares each
# row with the installed registry for every key that mflux already has.
# - qwen-image-edit-2511: the Qwen-Image edit family. Up to mflux 0.21 its two aliases sat on the
#   `qwen-image-edit` config (same edit pipeline, same Qwen-Image VAE); 0.22 gave it its own key
#   so it loads the 2511 weights instead of 2509.
# - krea-2-controlnet-depth: Krea 2 Turbo plus a depth ControlNet on the same transformer and VAE.
# - Qwen-Image 2.1 (Turbo, ControlNet union): the 64-channel 2.1 VAE taew2.1 cannot decode; its
#   in-loop latent has the same (1, N, 64) shape as Qwen-Image's, so a claim would decode garbage.
# - FLUX 3 Image Klein (and base): no shown equivalence with the FLUX.2 Klein latent taef2 reads.
# - Ming-Image 0.1 Design: no tiny decoder for it.
_MFLUX_0_23_NEW_KEYS: dict[str, tuple[str, list[str], str | None]] = {
    "qwen-image-edit-2511": (
        "Qwen/Qwen-Image-Edit-2511",
        ["qwen-image-edit-2511", "qwen-edit-2511"],
        "qwen-image",
    ),
    "krea-2-controlnet-depth": (
        "krea/Krea-2-Turbo",
        ["krea-2-controlnet-depth", "krea-2-controlnet", "krea2-controlnet"],
        "krea2",
    ),
    "qwen-image-2.1-turbo": (
        "Qwen/Qwen-Image-2.1-Turbo",
        ["qwen-image-2.1-turbo", "qwen-2.1-turbo", "qwen-image-21-turbo", "qwen-image-turbo-21"],
        None,
    ),
    "qwen-image-2.1-controlnet-union": (
        "Qwen/Qwen-Image-2.1",
        ["qwen-image-2.1-controlnet-union", "qwen-image-2.1-controlnet", "qwen-2.1-controlnet"],
        None,
    ),
    "qwen-image-2.1-turbo-controlnet-union": (
        "Qwen/Qwen-Image-2.1-Turbo",
        [
            "qwen-image-2.1-turbo-controlnet-union",
            "qwen-image-2.1-turbo-controlnet",
            "qwen-2.1-turbo-controlnet",
        ],
        None,
    ),
    # The FLUX 3 alias lists are mflux 0.24.0's; 0.24 appended `flux-3-image-klein(-base)`,
    # which _REGISTRY_ALIAS_INTRODUCED_IN dates so the 0.23 registry still compares equal.
    "flux3-image-klein": (
        "black-forest-labs/flux-3-image-klein",
        ["flux3-image-klein", "flux3-klein", "flux3", "flux-3-image-klein"],
        None,
    ),
    "flux3-image-klein-base": (
        "black-forest-labs/flux-3-image-klein-base",
        ["flux3-image-klein-base", "flux3-klein-base", "flux3-base", "flux-3-image-klein-base"],
        None,
    ),
    "ming-image-design": (
        "inclusionAI/Ming-Image-0.1-Design",
        ["ming-image-design", "ming-image", "ming-design", "ming"],
        None,
    ),
}

# Registry keys mflux 0.24 added, as mflux 0.24.0's `model_config.py` builds them (model_name,
# aliases in mflux's order; base_model is None for all), read 2026-10-28 from the installed
# source -> None: no tiny decoder in this library previews any of them, and no prefix or alias
# here may claim one (Iris, MOSS-Image, Anima, FIBO 2.0).
_MFLUX_0_24_NEW_KEYS: dict[str, tuple[str, list[str], str | None]] = {
    "iris-3b": ("speridlabs/iris-3b", ["iris-3b", "iris-3b-turbo", "iris-turbo", "iris"], None),
    "iris-3b-base": ("speridlabs/iris-3b", ["iris-3b-base", "iris-base"], None),
    "fibo2": ("briaai/Fibo-2.0-dev", ["fibo2", "fibo-2.0-dev"], None),
    "fibo2-edit-turbo": (
        "briaai/Fibo-2.0-edit-turbo",
        ["fibo2-edit-turbo", "fibo-2-edit-turbo", "fibo-2.0-edit-turbo"],
        None,
    ),
    "moss-image-sft": (
        "OpenMOSS-Team/MOSS-Image-SFT",
        ["moss-image-sft", "moss-image", "moss"],
        None,
    ),
    "moss-image-base": ("OpenMOSS-Team/MOSS-Image-Base", ["moss-image-base", "moss-base"], None),
    "anima-turbo": ("circlestone-labs/Anima", ["anima-turbo", "anima"], None),
    "anima-aesthetic": ("circlestone-labs/Anima", ["anima-aesthetic"], None),
    "anima-base": ("circlestone-labs/Anima", ["anima-base"], None),
}

# Every key mflux added after 0.20, with the strings mflux builds for it.
_MFLUX_NEW_KEYS = {**_MFLUX_0_23_NEW_KEYS, **_MFLUX_0_24_NEW_KEYS}


@pytest.mark.parametrize(
    ("model_name", "aliases", "expected"),
    list(_MFLUX_NEW_KEYS.values()),
    ids=list(_MFLUX_NEW_KEYS),
)
def test_an_mflux_0_21_to_0_24_model_resolves_to_its_kernel_or_is_rejected(
    model_name: str, aliases: list[str], expected: str | None
) -> None:
    """What `LivePreviewCallback(flux=model)` decides for each model mflux 0.21 to 0.24 added,
    from the config mflux builds for it. Catches: a family prefix widened so it claims a model
    with another latent (`Qwen/Qwen-Image` claiming `Qwen/Qwen-Image-2.1-Turbo`, a FLUX prefix
    claiming FLUX 3), which previews garbage instead of refusing before any weights load; or a
    prefix narrowed so the 2511 edit model or the Krea 2 ControlNet is refused."""
    config = _FakeModelConfig(model_name=model_name, aliases=aliases, base_model=None)
    if expected is None:
        with pytest.raises(UnsupportedMfluxModelError, match=model_name.replace(".", r"\.")):
            resolve_kernel_from_model_config(config)
    else:
        assert resolve_kernel_from_model_config(config) is KERNELS[expected]


@pytest.mark.parametrize(
    ("key", "alias", "expected"),
    [
        (key, alias, expected)
        for key, (_, aliases, expected) in _MFLUX_NEW_KEYS.items()
        for alias in aliases
    ],
)
def test_each_alias_of_an_mflux_0_21_to_0_24_model_resolves_alone(
    key: str, alias: str, expected: str | None
) -> None:
    """A local checkout of one of these models carries a filesystem path as its name, so only
    its alias can decide. Catches: a supported model's alias missing from its kernel (the Krea 2
    ControlNet's `krea-2-controlnet` refused), or a 2.1 / FLUX 3 / Ming alias registered on a
    kernel that cannot decode that latent."""
    config = _FakeModelConfig(model_name=f"/Users/me/models/{key}", aliases=[alias])
    if expected is None:
        with pytest.raises(UnsupportedMfluxModelError):
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


def test_a_mirror_of_qwen_image_2_1_is_rejected_not_claimed_by_qwen_image() -> None:
    """Catches: the `qwen-image` name prefix claiming Qwen-Image 2.1 (mflux 0.20), whose
    64-channel VAE is not the Wan 2.1 latent taew2.1 decodes. Nothing downstream would catch it:
    2.1's in-loop latent has the same (1, N, 64) shape as Qwen-Image's packed latent, so the
    preview would silently decode garbage."""
    config = _FakeModelConfig(
        model_name="someone/Qwen-Image-2.1-mflux-4bit",
        aliases=["qwen-image-2.1", "qwen-2.1", "qwen-image-21"],
        base_model="Qwen/Qwen-Image-2.1",
    )
    with pytest.raises(UnsupportedMfluxModelError, match=r"Qwen-Image-2\.1"):
        resolve_kernel_from_model_config(config)


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
    # mflux 0.20: new 64-channel VAE, no tiny decoder exists for it.
    "qwen-image-2.1": None,
    # mflux 0.21 to 0.24, classified in _MFLUX_0_23_NEW_KEYS and _MFLUX_0_24_NEW_KEYS above.
    **{key: expected for key, (_, _, expected) in _MFLUX_NEW_KEYS.items()},
}
# Registry keys that only exist from a given mflux release on; a key newer than the installed
# mflux is expected to be absent. The walk covers mflux 0.19 and newer; the declared floor is
# 0.24.
_REGISTRY_KEY_INTRODUCED_IN: dict[str, tuple[int, int]] = {
    "qwen-image-2.1": (0, 20),
    "ming-image-design": (0, 21),
    "qwen-image-edit-2511": (0, 22),
    "qwen-image-2.1-controlnet-union": (0, 22),
    "qwen-image-2.1-turbo": (0, 23),
    "qwen-image-2.1-turbo-controlnet-union": (0, 23),
    "flux3-image-klein": (0, 23),
    "flux3-image-klein-base": (0, 23),
    "krea-2-controlnet-depth": (0, 23),
    **dict.fromkeys(_MFLUX_0_24_NEW_KEYS, (0, 24)),
}
# Aliases mflux appended to a registry entry from a given release on; an older installed mflux
# builds the entry without them.
_REGISTRY_ALIAS_INTRODUCED_IN: dict[str, tuple[int, int]] = {
    "flux-3-image-klein": (0, 24),
    "flux-3-image-klein-base": (0, 24),
}


# The newest mflux the 0.21 to 0.24 tables were read from. On a newer mflux the entry guard
# skips: an edited entry there is bookkeeping for the lock move, and the drift walk still
# checks routing on every minor.
# Moving it also moves the (0, 24) literal in test_the_entry_guard_skips_only_on_an_mflux_newer_than_recorded.
_NEWEST_RECORDED_MFLUX_MINOR = (0, 24)


def _entry_guard_skip_reason(installed: tuple[int, int]) -> str | None:
    """Why the entry guard skips on `installed`, or None when it compares."""
    if installed <= _NEWEST_RECORDED_MFLUX_MINOR:
        return None
    major, minor = installed
    return (
        f"the 0.21 to 0.24 tables were read from mflux up to 0.24; installed mflux "
        f"{major}.{minor} is newer, so only the routing walk checks it"
    )


def _registry_aliases_as_of(aliases: list[str], installed: tuple[int, int]) -> list[str]:
    """The alias list an mflux at `installed` builds for a table row, in mflux's order."""
    return [a for a in aliases if _REGISTRY_ALIAS_INTRODUCED_IN.get(a, (0, 0)) <= installed]


def _installed_mflux_minor() -> tuple[int, int]:
    from importlib.metadata import version

    major, minor = version("mflux").split(".")[:2]
    return int(major), int(minor)


def _expected_registry_keys() -> set[str]:
    installed = _installed_mflux_minor()
    return {
        key
        for key in _EXPECTED_BY_REGISTRY_KEY
        if _REGISTRY_KEY_INTRODUCED_IN.get(key, (0, 0)) <= installed
    }


def test_expected_registry_keys_follow_the_installed_mflux_version(monkeypatch) -> None:
    """Catches: the version table being ignored, so the walk demands `qwen-image-2.1` from an
    mflux 0.19.x install (which never had it) or tolerates its absence on 0.20."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.19.2")
    assert "qwen-image-2.1" not in _expected_registry_keys()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.20.0")
    assert "qwen-image-2.1" in _expected_registry_keys()
    assert not set(_MFLUX_NEW_KEYS) & _expected_registry_keys()


@pytest.mark.parametrize(
    ("installed", "present"),
    [
        ("0.20.0", set()),
        ("0.21.0", {"ming-image-design"}),
        (
            "0.22.0",
            {"ming-image-design", "qwen-image-edit-2511", "qwen-image-2.1-controlnet-union"},
        ),
        ("0.23.0", set(_MFLUX_0_23_NEW_KEYS)),
        ("0.24.0", set(_MFLUX_0_23_NEW_KEYS) | set(_MFLUX_0_24_NEW_KEYS)),
    ],
)
def test_the_registry_walk_expects_each_new_key_from_its_mflux_release_on(
    monkeypatch: pytest.MonkeyPatch, installed: str, present: set[str]
) -> None:
    """The walk compares the installed registry with this map, so after the lock moves to a
    newer mflux it must demand exactly the keys that release has (Ming in 0.21; the 2511 edit
    key and the 2.1 ControlNet in 0.22; Turbo, its ControlNet, FLUX 3 and the Krea 2 ControlNet
    in 0.23). Catches: a new key left out of the map (the walk fails on the lock move), or
    dated to the wrong release (the walk fails on an mflux in the supported range)."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    assert set(_MFLUX_NEW_KEYS) & _expected_registry_keys() == present
    for key in present:
        assert _EXPECTED_BY_REGISTRY_KEY[key] == _MFLUX_NEW_KEYS[key][2], key


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ((0, 23), ["flux3-image-klein", "flux3-klein", "flux3"]),
        ((0, 24), ["flux3-image-klein", "flux3-klein", "flux3", "flux-3-image-klein"]),
        ((0, 25), ["flux3-image-klein", "flux3-klein", "flux3", "flux-3-image-klein"]),
    ],
)
def test_a_row_lists_an_appended_alias_only_from_the_mflux_release_that_added_it(
    installed: tuple[int, int], expected: list[str]
) -> None:
    """mflux 0.23.0 builds FLUX 3 Image Klein with three aliases, 0.24.0 with four
    (`flux-3-image-klein` appended). Catches: the 0.24 alias demanded from a 0.23 registry (the
    entry guard below fails on the lock's mflux), or dropped on 0.24 and later (the guard stops
    seeing the real list), or the order changed while filtering."""
    row_aliases = _MFLUX_NEW_KEYS["flux3-image-klein"][1]
    assert _registry_aliases_as_of(row_aliases, installed) == expected


@pytest.mark.parametrize(
    ("installed", "skips"),
    [((0, 23), False), ((0, 24), False), ((0, 25), True), ((1, 0), True)],
)
def test_the_entry_guard_compares_up_to_the_newest_recorded_mflux_and_skips_beyond(
    installed: tuple[int, int], skips: bool
) -> None:
    """The tables were read from mflux up to 0.24; a newer mflux may edit those entries
    without changing routing, which the drift walk checks on every minor. Catches: the guard
    still comparing on 0.25 (the weekly newest-mflux job red for bookkeeping), or skipping on
    0.24 itself (the lock's mflux no longer checked)."""
    reason = _entry_guard_skip_reason(installed)
    if skips:
        assert reason is not None
        assert f"installed mflux {installed[0]}.{installed[1]}" in reason
    else:
        assert reason is None


@pytest.mark.parametrize(
    "faked_version",
    [None, "0.25.0"],
    ids=["installed-mflux-compares", "faked-0.25-skips"],
)
def test_the_entry_guard_skips_only_on_an_mflux_newer_than_recorded(
    monkeypatch: pytest.MonkeyPatch, faked_version: str | None
) -> None:
    """The real guard against the installed registry. The compare case runs under the real
    installed version, never a faked one: a faked 0.24 over an older registry (0.23 or
    earlier) compares tables the registry cannot match. It runs only
    when the installed mflux is 0.24, the newest minor the tables were read from, and there a
    skip fails it. The faked 0.25 case only skips, so it holds on any registry.
    Catches: the guard skipping on 0.24 itself (the lock's mflux no longer checked, e.g. `<`
    for `<=` in the skip helper or the recorded minor set to 0.23), or the skip helper not
    wired into the guard (a 0.25 registry compared anyway)."""
    import importlib.metadata

    pytest.importorskip("mflux.models.common.config.model_config")
    if faked_version is None:
        installed = _installed_mflux_minor()
        # A literal, not _NEWEST_RECORDED_MFLUX_MINOR: a wrong recorded minor must not move it.
        if installed != (0, 24):
            pytest.skip(
                f"the entry guard compares only on mflux 0.24, the newest minor the tables "
                f"were read from; installed mflux is {installed[0]}.{installed[1]}"
            )
        # A skip here would only skip this test too; on the recorded minor it is the bug.
        try:
            test_the_0_21_to_0_24_tables_match_the_installed_mflux_registry_entry_for_entry()
        except pytest.skip.Exception as skipped:
            pytest.fail(f"the entry guard skipped on the installed mflux 0.24: {skipped}")
    else:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: faked_version)
        with pytest.raises(pytest.skip.Exception, match=r"installed mflux 0\.25 is newer"):
            test_the_0_21_to_0_24_tables_match_the_installed_mflux_registry_entry_for_entry()


def test_the_0_21_to_0_24_tables_match_the_installed_mflux_registry_entry_for_entry() -> None:
    """The tables above claim to copy mflux's registry. For every key the installed mflux
    already has, its `model_name`, alias list (order included, the key-named alias first, minus
    aliases a later mflux appended) and `base_model` must equal the installed `AVAILABLE_MODELS`
    entry, so the alias tests exercise the strings mflux really builds. Catches: an alias
    dropped or reordered in a table (the key-named alias was missing from six rows), or a
    model_name typo. Keys newer than the installed mflux are not checked; on mflux 0.19 (older
    than the 0.24 floor) there is nothing to compare. This guard is bookkeeping, not routing, so it skips on an
    mflux newer than the tables (0.24); the routing walk below covers every minor."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    installed = _installed_mflux_minor()
    skip_reason = _entry_guard_skip_reason(installed)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    keys = [k for k in _MFLUX_NEW_KEYS if _REGISTRY_KEY_INTRODUCED_IN[k] <= installed]
    if not keys:
        pytest.skip(f"no key of the 0.21 to 0.24 tables exists in the installed mflux {installed}")

    mismatches = []
    for key in keys:
        model_name, aliases, _ = _MFLUX_NEW_KEYS[key]
        expected = (model_name, _registry_aliases_as_of(aliases, installed), None)
        entry = mflux_config.AVAILABLE_MODELS.get(key)
        if entry is None:
            mismatches.append((key, "not in the installed registry", [], None))
        elif (entry.model_name, list(entry.aliases), entry.base_model) != expected:
            mismatches.append((key, entry.model_name, list(entry.aliases), entry.base_model))
    assert mismatches == []


# Alias strings kept from releases before mflux registered its own; not in AVAILABLE_MODELS.
_LEGACY_ALIASES = {"flux1", "flux-dev", "flux-schnell", "flux2"}


# --- The registry walk, split by what a change means --------------------------------------
# For every key of the installed registry, the walk resolves the config mflux builds (the prefix
# arms decide for it) and each alias alone under a local-path name (only the alias arm can
# decide). A key in _EXPECTED_BY_REGISTRY_KEY must keep its kernel, or keep being refused. A key
# not in the table is "unclassified": if every probe is refused it is listed in a warning and
# the walk stays green; if any probe lands in a kernel the walk fails naming key, probe and
# kernel, since some prefix or alias now routes a model nobody classified. A key the table
# expects on the installed mflux but the registry no longer has fails as well: its rows would
# keep exercising strings mflux no longer builds, and a renamed model would hide behind it.


class UnclassifiedMfluxModelsWarning(UserWarning):
    """mflux registry keys this file has not classified; the resolver refuses every one."""


class _RoutingChange(NamedTuple):
    """One probe of one registry key that did not route as classified. `expected` is the
    table's kernel name (None: refused); for an unclassified key (`known=False`) it is None."""

    key: str
    probe: str  # "config", or "alias '<alias>'" for an alias resolved alone
    expected: str | None
    got: str | None  # the kernel name it resolved to, None when refused
    known: bool

    def __str__(self) -> str:
        if not self.known:
            return f"{self.key} [{self.probe}]: unclassified key resolves to {self.got}"
        return (
            f"{self.key} [{self.probe}]: expected {self.expected or 'refused'}, "
            f"got {self.got or 'refused'}"
        )


@dataclass
class _RegistryDrift:
    unclassified: list[str]  # unknown keys every probe of which is refused, sorted
    changed: list[_RoutingChange]  # in registry order, config probe first
    removed: list[str]  # expected on the installed mflux but not registered, sorted


def _resolved_kernel_name(config: Any) -> str | None:
    try:
        return resolve_kernel_from_model_config(config).name
    except UnsupportedMfluxModelError:
        return None


def _registry_drift(
    available: Mapping[str, Any],
    *,
    expected: Mapping[str, str | None],
    expected_keys: set[str],
) -> _RegistryDrift:
    """Resolve every registry entry's config and each of its aliases alone, and sort the result
    into unclassified keys, routing changes and removed keys (see the section comment)."""
    unclassified: list[str] = []
    changed: list[_RoutingChange] = []
    for key, config in available.items():
        probes = [("config", config)] + [
            (f"alias {alias!r}", _FakeModelConfig(f"/Users/me/models/{key}", aliases=[alias]))
            for alias in config.aliases
        ]
        known = key in expected
        want = expected.get(key)
        key_changes = [
            _RoutingChange(key, probe, want, got, known)
            for probe, probe_config in probes
            if (got := _resolved_kernel_name(probe_config)) != want
        ]
        if not known and not key_changes:
            unclassified.append(key)
        changed.extend(key_changes)
    return _RegistryDrift(
        unclassified=sorted(unclassified),
        changed=changed,
        removed=sorted(expected_keys - set(available)),
    )


def test_the_split_lists_an_unknown_refused_key_without_a_routing_change() -> None:
    """Catches: the walk failing (or staying silent) on a new mflux model no kernel claims,
    the case that used to turn the weekly newest-mflux job red for every model mflux added."""
    available = {"acme-new": _FakeModelConfig("acme/NewModel-1", aliases=["acme-new", "acme"])}

    drift = _registry_drift(available, expected={}, expected_keys=set())

    assert drift == _RegistryDrift(unclassified=["acme-new"], changed=[], removed=[])


def test_the_split_fails_an_unknown_key_a_family_prefix_claims() -> None:
    """Catches: an unclassified key treated as harmless although a broad prefix routes it to a
    kernel (a FLUX.1 fine-tune that would get taef1 without anyone deciding)."""
    available = {
        "acme-flux1": _FakeModelConfig("black-forest-labs/FLUX.1-acme", aliases=["acme-flux1"])
    }

    drift = _registry_drift(available, expected={}, expected_keys=set())

    assert drift.unclassified == []
    assert drift.changed == [_RoutingChange("acme-flux1", "config", None, "taef1", known=False)]


def test_the_split_fails_an_unknown_key_one_alias_alone_routes() -> None:
    """An unclassified key whose name no prefix claims but one of whose aliases is a kernel's
    alias: the full config resolves through the alias arm, and the failure names the alias that
    does it. Catches: aliases not probed alone (the report cannot say which alias routes), or a
    refused alias reported as a change."""
    available = {"acme-zimage": _FakeModelConfig("acme/Img-1", aliases=["acme-zimage", "zimage"])}

    drift = _registry_drift(available, expected={}, expected_keys=set())

    assert drift.unclassified == []
    assert drift.changed == [
        _RoutingChange("acme-zimage", "config", None, "zimage", known=False),
        _RoutingChange("acme-zimage", "alias 'zimage'", None, "zimage", known=False),
    ]


def test_the_split_fails_a_known_key_whose_kernel_changed() -> None:
    """Catches: a classified key whose kernel moved (FLUX.1-dev now claimed by the Klein kernel,
    or a kernel dropping `dev` from its aliases) passing as long as it resolves to something."""
    available = {"dev": _FakeModelConfig("black-forest-labs/FLUX.1-dev", aliases=["dev"])}

    drift = _registry_drift(available, expected={"dev": "taef2"}, expected_keys={"dev"})

    assert drift.unclassified == []
    assert drift.changed == [
        _RoutingChange("dev", "config", "taef2", "taef1", known=True),
        _RoutingChange("dev", "alias 'dev'", "taef2", "taef1", known=True),
    ]


def test_the_split_fails_a_known_refused_key_a_kernel_now_claims() -> None:
    """Catches: a key classified as unsupported (None) passing whatever it resolves to."""
    available = {"lens-turbo": _FakeModelConfig("black-forest-labs/FLUX.1-lens", aliases=[])}

    drift = _registry_drift(available, expected={"lens-turbo": None}, expected_keys={"lens-turbo"})

    assert drift.changed == [_RoutingChange("lens-turbo", "config", None, "taef1", known=True)]


def test_the_split_accepts_known_keys_that_route_as_classified() -> None:
    """Catches: a walk that reports a change, or lists a known key as unclassified, when every
    key keeps its kernel or its refusal."""
    available = {
        "dev": _FakeModelConfig("black-forest-labs/FLUX.1-dev", aliases=["dev"]),
        "fibo": _FakeModelConfig("briaai/FIBO", aliases=["fibo"]),
    }

    drift = _registry_drift(
        available, expected={"dev": "taef1", "fibo": None}, expected_keys={"dev", "fibo"}
    )

    assert drift == _RegistryDrift(unclassified=[], changed=[], removed=[])


def test_the_split_fails_a_key_mflux_dropped() -> None:
    """Catches: a classified key mflux no longer registers going unnoticed (only keys the
    installed mflux should have count; a key newer than it is not "removed")."""
    available = {"dev": _FakeModelConfig("black-forest-labs/FLUX.1-dev", aliases=["dev"])}

    # "flux9" is classified but dated after the installed mflux, so it is not in expected_keys.
    expected = {"dev": "taef1", "schnell": "taef1", "flux9": None}

    drift = _registry_drift(available, expected=expected, expected_keys={"dev", "schnell"})

    assert drift.removed == ["schnell"]


def test_a_routing_change_message_names_the_key_the_probe_and_the_kernel() -> None:
    """Catches: a failure that says "drift" without the key, probe and kernel someone needs
    to classify the model or revert the routing change."""
    unknown = _RoutingChange("acme-flux1", "config", None, "taef1", known=False)
    known = _RoutingChange("dev", "alias 'dev'", "taef1", None, known=True)

    assert str(unknown) == "acme-flux1 [config]: unclassified key resolves to taef1"
    assert str(known) == "dev [alias 'dev']: expected taef1, got refused"


def test_every_installed_mflux_registry_entry_routes_as_classified() -> None:
    """Catches: a routing change between this library and the installed mflux registry: a
    model this file has not classified that a family prefix now claims, a classified model
    whose config resolves to another kernel (or is newly refused), or a model mflux dropped
    that the table still expects. Unclassified models the resolver refuses are listed in an
    UnclassifiedMfluxModelsWarning instead of failing. The alias arm has its own test below."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")

    drift = _registry_drift(
        mflux_config.AVAILABLE_MODELS,
        expected=_EXPECTED_BY_REGISTRY_KEY,
        expected_keys=_expected_registry_keys(),
    )

    if drift.unclassified:
        warnings.warn(
            UnclassifiedMfluxModelsWarning(
                "unclassified mflux registry keys (refused by the resolver): "
                + ", ".join(drift.unclassified)
            ),
            stacklevel=1,
        )
    config_changes = [str(c) for c in drift.changed if c.probe == "config"]
    assert not config_changes, "routing changed; classify the key or fix the kernel:\n" + "\n".join(
        config_changes
    )
    assert not drift.removed, "mflux dropped these keys; remove them from the tables: " + ", ".join(
        drift.removed
    )


def test_every_registered_alias_resolves_on_its_own_as_classified() -> None:
    """Catches: a kernel dropping one of its aliases (`qwen-2512`), which nothing else sees:
    mflux-built configs resolve through the prefix arms, and a config carrying the full alias
    list still matches on the others. Each alias is resolved alone, with a name no prefix
    claims, so the alias arm has to carry it. An alias of an unclassified key that a kernel
    claims fails here too; an unclassified key whose aliases are all refused does not."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")

    drift = _registry_drift(
        mflux_config.AVAILABLE_MODELS,
        expected=_EXPECTED_BY_REGISTRY_KEY,
        expected_keys=_expected_registry_keys(),
    )

    alias_changes = [str(c) for c in drift.changed if c.probe != "config"]
    assert not alias_changes, "routing changed; classify the key or fix the kernel:\n" + "\n".join(
        alias_changes
    )


def test_the_walk_warns_and_stays_green_for_an_unknown_refused_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real-registry tests with one fake model added to the installed AVAILABLE_MODELS.
    Catches: the warning not raised (the listing vanishes from the CI log), its message changed
    so log searches miss it, or either walk failing (or raising KeyError) on the unknown key."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    fake = _FakeModelConfig("acme/NewModel-1", aliases=["acme-new", "acme"])
    monkeypatch.setitem(mflux_config.AVAILABLE_MODELS, "acme-new", fake)

    with pytest.warns(UnclassifiedMfluxModelsWarning) as record:
        test_every_installed_mflux_registry_entry_routes_as_classified()
    test_every_registered_alias_resolves_on_its_own_as_classified()

    # One warning; the installed mflux may list real unclassified keys beside the fake one.
    assert len(record) == 1
    prefix = "unclassified mflux registry keys (refused by the resolver): "
    message = str(record[0].message)
    assert message.startswith(prefix), message
    assert "acme-new" in message.removeprefix(prefix).split(", ")


# The three failure checks below drive the config walk, which also warns about any real
# unclassified key; that listing belongs to the walk's own test, so it is not repeated here.
_IGNORE_UNCLASSIFIED = pytest.mark.filterwarnings("ignore:unclassified mflux registry keys")


@_IGNORE_UNCLASSIFIED
def test_the_walk_fails_for_an_unknown_key_a_prefix_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: the real-registry walk swallowing a routing change on an unclassified key."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    fake = _FakeModelConfig("black-forest-labs/FLUX.1-acme", aliases=["acme-flux1"])
    monkeypatch.setitem(mflux_config.AVAILABLE_MODELS, "acme-flux1", fake)

    with pytest.raises(AssertionError, match=r"acme-flux1 \[config\]: unclassified .* taef1"):
        test_every_installed_mflux_registry_entry_routes_as_classified()


def test_the_walk_fails_for_an_unknown_key_whose_alias_a_kernel_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: the real-registry alias walk skipping unclassified keys."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    fake = _FakeModelConfig("acme/Img-1", aliases=["acme-zimage", "zimage"])
    monkeypatch.setitem(mflux_config.AVAILABLE_MODELS, "acme-zimage", fake)

    with pytest.raises(AssertionError, match=r"acme-zimage \[alias 'zimage'\]: .* zimage"):
        test_every_registered_alias_resolves_on_its_own_as_classified()


@_IGNORE_UNCLASSIFIED
def test_the_walk_fails_for_a_key_mflux_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches: the real-registry walk computing removed keys but never failing on them
    (`schnell` taken out of the installed AVAILABLE_MODELS while the table still expects it)."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    monkeypatch.delitem(mflux_config.AVAILABLE_MODELS, "schnell")

    with pytest.raises(AssertionError, match=r"mflux dropped these keys.*: schnell"):
        test_every_installed_mflux_registry_entry_routes_as_classified()


@_IGNORE_UNCLASSIFIED
def test_the_walk_fails_when_a_known_key_is_expected_on_another_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: the real-registry walk passing a classified key whatever it resolves to (here
    the table is flipped so `dev`, which resolves to taef1, is expected on taef2)."""
    pytest.importorskip("mflux.models.common.config.model_config")
    monkeypatch.setitem(_EXPECTED_BY_REGISTRY_KEY, "dev", "taef2")

    with pytest.raises(AssertionError, match=r"dev \[config\]: expected taef2, got taef1"):
        test_every_installed_mflux_registry_entry_routes_as_classified()
    with pytest.raises(AssertionError, match=r"dev \[alias 'dev'\]: expected taef2, got taef1"):
        test_every_registered_alias_resolves_on_its_own_as_classified()


# Kernel aliases mflux registers only from a given release on (the Krea 2 depth ControlNet, 0.23);
# an older installed mflux does not know them yet, so they are not stale there.
_KERNEL_ALIAS_INTRODUCED_IN: dict[str, tuple[int, int]] = {
    "krea-2-controlnet-depth": (0, 23),
    "krea-2-controlnet": (0, 23),
    "krea2-controlnet": (0, 23),
}


def _kernel_aliases_newer_than_installed() -> set[str]:
    installed = _installed_mflux_minor()
    return {a for a, since in _KERNEL_ALIAS_INTRODUCED_IN.items() if since > installed}


def test_kernel_aliases_from_a_newer_mflux_are_exempt_only_before_that_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches: the exemption outliving its release, so on mflux 0.23 a Krea 2 ControlNet
    alias mflux renamed or dropped would still pass the stale-alias check below."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.22.1")
    assert _kernel_aliases_newer_than_installed() == {
        "krea-2-controlnet-depth",
        "krea-2-controlnet",
        "krea2-controlnet",
    }
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.23.0")
    assert _kernel_aliases_newer_than_installed() == set()


def test_every_kernel_alias_is_one_mflux_registers() -> None:
    """Catches: a kernel alias list going stale (mflux renames `zimage-turbo`); the registry
    walk cannot see it because prefixes decide first, and a dead alias is a silent lie about
    what the library supports."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    registered = {a.lower() for c in mflux_config.AVAILABLE_MODELS.values() for a in c.aliases}
    not_yet_registered = _kernel_aliases_newer_than_installed()

    for kernel in KERNELS.values():
        if kernel.integration is None:
            continue
        stale = (
            set(kernel.integration.mflux_models) - registered - _LEGACY_ALIASES - not_yet_registered
        )
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
