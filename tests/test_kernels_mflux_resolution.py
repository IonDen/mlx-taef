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
    "flux3-image-klein": (
        "black-forest-labs/flux-3-image-klein",
        ["flux3-image-klein", "flux3-klein", "flux3"],
        None,
    ),
    "flux3-image-klein-base": (
        "black-forest-labs/flux-3-image-klein-base",
        ["flux3-image-klein-base", "flux3-klein-base", "flux3-base"],
        None,
    ),
    "ming-image-design": (
        "inclusionAI/Ming-Image-0.1-Design",
        ["ming-image-design", "ming-image", "ming-design", "ming"],
        None,
    ),
}


@pytest.mark.parametrize(
    ("model_name", "aliases", "expected"),
    list(_MFLUX_0_23_NEW_KEYS.values()),
    ids=list(_MFLUX_0_23_NEW_KEYS),
)
def test_an_mflux_0_23_model_resolves_to_its_kernel_or_is_rejected(
    model_name: str, aliases: list[str], expected: str | None
) -> None:
    """What `LivePreviewCallback(flux=model)` decides for each model mflux 0.21 to 0.23 added,
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
        for key, (_, aliases, expected) in _MFLUX_0_23_NEW_KEYS.items()
        for alias in aliases
    ],
)
def test_each_alias_of_an_mflux_0_23_model_resolves_alone(
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
    # mflux 0.21 to 0.23, classified in _MFLUX_0_23_NEW_KEYS above.
    **{key: expected for key, (_, _, expected) in _MFLUX_0_23_NEW_KEYS.items()},
}
# Registry keys that only exist from a given mflux release on; a key newer than the installed
# mflux is expected to be absent. The walk covers mflux 0.19 and newer, the declared floor.
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
}


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
    assert not set(_MFLUX_0_23_NEW_KEYS) & _expected_registry_keys()


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
    assert set(_MFLUX_0_23_NEW_KEYS) & _expected_registry_keys() == present
    for key in present:
        assert _EXPECTED_BY_REGISTRY_KEY[key] == _MFLUX_0_23_NEW_KEYS[key][2], key


def test_the_0_23_table_matches_the_installed_mflux_registry_entry_for_entry() -> None:
    """The table above claims to copy mflux's registry. For every key the installed mflux
    already has, its `model_name`, alias list (order included, the key-named alias first) and
    `base_model` must equal the installed `AVAILABLE_MODELS` entry, so the alias tests exercise
    the strings mflux really builds. Catches: an alias dropped or reordered in the table (the
    key-named alias was missing from six rows), or a model_name typo. Keys newer than the
    installed mflux are not checked; on the 0.19 floor there is nothing to compare."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    installed = _installed_mflux_minor()
    keys = [k for k in _MFLUX_0_23_NEW_KEYS if _REGISTRY_KEY_INTRODUCED_IN[k] <= installed]
    if not keys:
        pytest.skip(f"no key of the 0.23 table exists in the installed mflux {installed}")

    mismatches = []
    for key in keys:
        model_name, aliases, _ = _MFLUX_0_23_NEW_KEYS[key]
        entry = mflux_config.AVAILABLE_MODELS[key]
        if (entry.model_name, list(entry.aliases), entry.base_model) != (model_name, aliases, None):
            mismatches.append((key, entry.model_name, list(entry.aliases), entry.base_model))
    assert mismatches == []


# Alias strings kept from releases before mflux registered its own; not in AVAILABLE_MODELS.
_LEGACY_ALIASES = {"flux1", "flux-dev", "flux-schnell", "flux2"}


def test_every_installed_mflux_registry_entry_is_classified() -> None:
    """Catches: drift between this library and the installed mflux registry in either
    direction: a model mflux added that a broad family prefix now claims (or rejects) without
    anyone deciding, or a model mflux dropped that the map still lists. mflux-built configs
    carry an owner-qualified name, so this walks the prefix arms; the alias arm is covered by
    the two alias tests below."""
    mflux_config = pytest.importorskip("mflux.models.common.config.model_config")
    if _installed_mflux_minor() < (0, 19):
        pytest.skip("the registry walk covers mflux 0.19 and newer")
    available = mflux_config.AVAILABLE_MODELS

    assert set(available) == _expected_registry_keys(), (
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
    if _installed_mflux_minor() < (0, 19):
        pytest.skip("the registry walk covers mflux 0.19 and newer")

    for key, config in mflux_config.AVAILABLE_MODELS.items():
        expected = _EXPECTED_BY_REGISTRY_KEY[key]
        for alias in config.aliases:
            local = _FakeModelConfig(model_name=f"/Users/me/models/{key}", aliases=[alias])
            if expected is None:
                with pytest.raises(UnsupportedMfluxModelError):
                    resolve_kernel_from_model_config(local)
            else:
                assert resolve_kernel_from_model_config(local) is KERNELS[expected], alias


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
