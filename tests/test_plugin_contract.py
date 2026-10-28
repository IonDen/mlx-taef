"""The slice of mlx-taef's public API that the `mflux-taef` plugin calls, pinned by signature and behaviour.

The plugin is planned at https://github.com/mflux-community/mflux-plugins (folder `taef/`, PyPI `mflux-taef`,
module `mflux.extras.taef`). It will depend on `mlx-taef[mflux]`, so a later mlx-taef release reaches installed
plugins through `pip install -U`. Before loading the mflux model it resolves the preview kernel with
`mlx_taef.kernels.resolve_kernel_from_model_config`, refuses unsupported models by catching
`UnsupportedMfluxModelError`, and builds `LivePreviewCallback(variant=..., every=..., save_to=...,
numbered_frames=..., on_error="disable")` without a model; then, per seed, it assigns `preview.save_to` and runs
the generation.

A failure here means a change would break mflux-taef users. A change here needs a matching mflux-taef release, or
a deprecation path that keeps the old shape working, before the mlx-taef release that carries it ships.

Expected values are literals from the plugin's design, not read back from the code under test. Weights come from
the committed converted decoders (`tests/converted/`), so nothing downloads. The hook-dispatch guards (every
before-loop / in-loop keyword plus the `**` catch-alls, driven through mflux's real `GenerationContext`) live in
`tests/test_integrations_mflux.py` and are not repeated here.
"""

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_taef
from mlx_taef import errors, kernels
from mlx_taef.kernels import resolve_kernel_from_model_config

P = inspect.Parameter
CONVERTED = Path(__file__).parent / "converted"

# variant name the plugin passes -> (API class name, committed decoder that stands in for the download)
_PLUGIN_VARIANTS: dict[str, tuple[str, str]] = {
    "taef1": ("TAEF1", "taef1_decoder.safetensors"),
    "taef2": ("TAEF2", "taef2_decoder.safetensors"),
    "zimage": ("ZImage", "taef1_decoder.safetensors"),
    "qwen-image": ("QwenImage", "qwen-image_decoder.safetensors"),
    "krea2": ("Krea2", "qwen-image_decoder.safetensors"),
}


def _callback_class() -> type:
    pytest.importorskip("mflux")
    from mlx_taef.integrations.mflux import LivePreviewCallback

    return LivePreviewCallback


def _offline_decoder(monkeypatch: pytest.MonkeyPatch, variant: str) -> object:
    """Point the variant's `from_pretrained` (the download boundary) at the committed decoder."""
    class_name, filename = _PLUGIN_VARIANTS[variant]
    cls = getattr(mlx_taef, class_name)
    real = cls.from_pretrained_local(CONVERTED / filename)
    monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, **kw: real))
    return real


def _zimage_step(cb: object) -> None:
    """One mflux Z-Image in-loop call: the latent is (16, 1, h, w) and needs no Config."""
    latent = mx.random.normal((16, 1, 8, 8), key=mx.random.key(0))
    cb.call_in_loop(t=0, seed=0, prompt="p", latents=latent, config=None, time_steps=None)  # type: ignore[attr-defined]


@dataclass
class _ModelConfigShaped:
    """The attributes of mflux's ModelConfig that the resolver reads; no mflux import needed."""

    model_name: str
    aliases: list[str] = field(default_factory=list)
    base_model: str | None = None


# LivePreviewCallback.__init__
# ---------------------------------------------------------------------------


def test_live_preview_keywords_the_plugin_passes_keep_their_names_and_kinds() -> None:
    """Red if any of the six keywords is renamed or loses its keyword-only position, or `flux` stops defaulting
    to None.

    Breaks: the plugin builds `LivePreviewCallback(variant=..., every=..., save_to=..., numbered_frames=...,
    on_error="disable")` by keyword with `flux` left out; a rename raises TypeError at startup, and with `flux`
    no longer defaulting to None the call raises TypeError or no longer means "no model". The plugin passes the other
    five, so their defaults are not its concern (see the next test). A change here needs a matching mflux-taef
    release or a deprecation path before the mlx-taef release ships."""
    params = inspect.signature(_callback_class().__init__).parameters
    pinned = ["flux", "variant", "every", "save_to", "numbered_frames", "on_error"]
    assert [(n, params[n].kind) for n in pinned if n in params] == [
        ("flux", P.KEYWORD_ONLY),
        ("variant", P.KEYWORD_ONLY),
        ("every", P.KEYWORD_ONLY),
        ("save_to", P.KEYWORD_ONLY),
        ("numbered_frames", P.KEYWORD_ONLY),
        ("on_error", P.KEYWORD_ONLY),
    ]
    assert params["flux"].default is None


def test_live_preview_defaults_for_unflagged_direct_runs_stay_put() -> None:
    """Red if the default of `every`, `save_to`, `numbered_frames` or `on_error` moves.

    Breaks: this guards code that builds `LivePreviewCallback(...)` directly and leaves these keywords out, not
    the plugin (which passes all four). A moved default silently changes such a run: previews at another step
    interval, written to another file, numbered instead of overwritten, or raising instead of warning once."""
    params = inspect.signature(_callback_class().__init__).parameters
    assert {n: params[n].default for n in ["every", "save_to", "numbered_frames", "on_error"]} == {
        "every": 5,
        "save_to": "preview.png",
        "numbered_frames": False,
        "on_error": "disable",
    }


@pytest.mark.parametrize("variant", ["zimage", "qwen-image", "krea2", "taef1", "taef2"])
def test_each_plugin_variant_builds_a_callback_without_a_model(
    variant: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red if a variant name the plugin passes is renamed or dropped from the callback's table.

    Breaks: the plugin builds the preview before the model loads, with `variant=` set from the resolved kernel
    and no `flux`; a missing name exits the command with "could not prepare the TAEF preview decoder". A change
    here needs a matching mflux-taef release or a deprecation path before the mlx-taef release ships."""
    decoder = _offline_decoder(monkeypatch, variant)
    cb = _callback_class()(
        variant=variant,
        every=2,
        save_to=tmp_path / "p.png",
        numbered_frames=False,
        on_error="disable",
    )
    assert cb.variant == variant
    assert cb.model is decoder


def test_numbered_frames_forces_a_preview_at_every_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red if `numbered_frames=True` stops overriding `every` (for example `self.every = every`).

    Breaks: the plugin's `--taef-frames` refuses `--taef-every` and promises one frame per step; with the
    override gone a frames run would keep every third step only. A change here needs a matching mflux-taef
    release or a deprecation path before the mlx-taef release ships."""
    _offline_decoder(monkeypatch, "zimage")
    cb = _callback_class()(
        variant="zimage", every=3, numbered_frames=True, save_to=tmp_path / "seed.png"
    )
    for _ in range(3):
        _zimage_step(cb)

    assert cb.every == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "seed_step00.png",
        "seed_step01.png",
        "seed_step02.png",
    ]


# save_to: assigned per seed after construction
# ---------------------------------------------------------------------------


def test_reassigned_save_to_receives_the_next_single_frame_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red if the target path is captured at construction or from an earlier write instead of read at write time.

    Breaks: the plugin builds one callback and sets `preview.save_to` to each seed's path before that seed's
    generation; seed 2's preview would overwrite seed 1's file. A change here needs a matching mflux-taef release
    or a deprecation path before the mlx-taef release ships."""
    _offline_decoder(monkeypatch, "zimage")
    first, second = tmp_path / "img_1.preview.png", tmp_path / "img_2.preview.png"
    cb = _callback_class()(variant="zimage", every=1, save_to=first)
    cb.call_before_loop(seed=1, prompt="p", latents=mx.zeros((16, 1, 8, 8)), config=None)
    _zimage_step(cb)

    cb.save_to = second
    cb.call_before_loop(seed=2, prompt="p", latents=mx.zeros((16, 1, 8, 8)), config=None)
    _zimage_step(cb)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["img_1.preview.png", "img_2.preview.png"]
    assert cb.saved_paths == [second]


def test_reassigned_save_to_names_the_next_numbered_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red if numbered frames are named from the construction-time path instead of the current `save_to`.

    Breaks: `--taef-frames` with several seeds sets a `{seed}` path per seed; frames must land as
    `<stem>_stepNN<suffix>` next to the new path, or seed 2's frames overwrite seed 1's. A change here needs a
    matching mflux-taef release or a deprecation path before the mlx-taef release ships."""
    _offline_decoder(monkeypatch, "zimage")
    cb = _callback_class()(variant="zimage", numbered_frames=True, save_to=tmp_path / "s1.png")
    cb.call_before_loop(seed=1, prompt="p", latents=mx.zeros((16, 1, 8, 8)), config=None)
    _zimage_step(cb)
    _zimage_step(cb)

    cb.save_to = tmp_path / "frames" / "s2.webp"
    cb.call_before_loop(seed=2, prompt="p", latents=mx.zeros((16, 1, 8, 8)), config=None)
    _zimage_step(cb)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "frames",
        "s1_step00.png",
        "s1_step01.png",
    ]
    assert [p.name for p in (tmp_path / "frames").iterdir()] == ["s2_step00.webp"]
    assert cb.saved_paths == [tmp_path / "frames" / "s2_step00.webp"]


# resolve_kernel_from_model_config
# ---------------------------------------------------------------------------


def test_resolver_is_public_and_takes_one_positional_argument() -> None:
    """Red if the resolver leaves `mlx_taef.kernels.__all__`, its first parameter turns keyword-only or gains a
    default, or it gains a second required parameter. An added optional parameter stays green.

    Breaks: the plugin calls `resolve_kernel_from_model_config(model_config)` before load to refuse a model with
    no preview decoder. A change here needs a matching mflux-taef release or a deprecation path before the
    mlx-taef release ships."""
    assert "resolve_kernel_from_model_config" in kernels.__all__
    first, *rest = inspect.signature(resolve_kernel_from_model_config).parameters.values()
    assert first.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD), first
    assert first.default is P.empty, first
    required = [
        p for p in rest if p.default is P.empty and p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)
    ]
    assert required == [], required


@pytest.fixture
def no_weights_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every download and weight-load path raise, so a resolver that touches one goes red."""

    def _refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("weights or network touched")

    import huggingface_hub

    import mlx_taef.download

    monkeypatch.setattr(mlx_taef.download, "get_or_convert", _refuse)
    monkeypatch.setattr(mlx_taef.Taef, "from_pretrained", classmethod(_refuse))
    monkeypatch.setattr(mlx_taef.Taef, "from_pretrained_local", classmethod(_refuse))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _refuse)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _refuse)
    monkeypatch.setattr(mx, "load", _refuse)


@pytest.mark.parametrize(
    ("model_name", "aliases", "variant"),
    [
        ("Tongyi-MAI/Z-Image", ["z-image", "zimage"], "zimage"),
        ("Tongyi-MAI/Z-Image-Turbo", ["z-image-turbo", "zimage-turbo"], "zimage"),
        ("Qwen/Qwen-Image-2512", ["qwen-image", "qwen", "qwen-image-2512"], "qwen-image"),
        ("Qwen/Qwen-Image-Edit-2511", ["qwen-image-edit-2511", "qwen-edit-2511"], "qwen-image"),
        ("krea/Krea-2-Turbo", ["krea-2", "krea2"], "krea2"),
    ],
)
@pytest.mark.usefixtures("no_weights_no_network")
def test_resolver_names_the_variant_the_callback_takes_without_loading_anything(
    model_name: str, aliases: list[str], variant: str
) -> None:
    """Red if a plugin family stops resolving, resolves to another kernel name, or resolution loads weights.

    Breaks: the plugin passes `resolve_kernel_from_model_config(model_config).name` straight to
    `LivePreviewCallback(variant=...)` and runs this check before the model load, offline. A change here needs a
    matching mflux-taef release or a deprecation path before the mlx-taef release ships."""
    kernel = resolve_kernel_from_model_config(_ModelConfigShaped(model_name, aliases))
    assert kernel.name == variant


@pytest.mark.parametrize(
    "model_config",
    [_ModelConfigShaped("Comfy-Org/Lens", ["lens-turbo", "lens"]), object()],
    ids=["unsupported-model", "no-model_name"],
)
@pytest.mark.usefixtures("no_weights_no_network")
def test_resolver_refuses_with_the_error_the_plugin_catches(model_config: object) -> None:
    """Red if an unsupported model or a config without `model_name` raises anything but UnsupportedMfluxModelError.

    Breaks: the plugin turns exactly this error into "TAEF has no preview decoder for <model>. Run <plain> ..."
    with exit 2 before load; any other exception escapes as a traceback. A change here needs a matching
    mflux-taef release or a deprecation path before the mlx-taef release ships."""
    with pytest.raises(errors.UnsupportedMfluxModelError):
        resolve_kernel_from_model_config(model_config)


# UnsupportedMfluxModelError
# ---------------------------------------------------------------------------


def test_unsupported_model_error_import_paths_and_bases() -> None:
    """Red if the error leaves the package root or `mlx_taef.errors`, or drops TaefError or LookupError as a base.

    Breaks: the plugin imports the named class from either path and catches it by that name, so a moved import
    path fails the plugin at import. The TaefError and LookupError bases are a public-API pin for users who
    catch the package's base error or a plain `LookupError` around model resolution; a lost base lets the
    refusal escape their handler. A change here needs a matching mflux-taef release or a deprecation path
    before the mlx-taef release ships."""
    assert mlx_taef.UnsupportedMfluxModelError is errors.UnsupportedMfluxModelError
    assert "UnsupportedMfluxModelError" in mlx_taef.__all__
    assert issubclass(errors.UnsupportedMfluxModelError, errors.TaefError)
    assert issubclass(errors.UnsupportedMfluxModelError, LookupError)
