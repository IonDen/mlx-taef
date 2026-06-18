"""Tests for the mflux integration (skipped if mflux not installed)."""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mflux = pytest.importorskip("mflux")

from mlx_taef.integrations.mflux import LivePreviewCallback, unpack_flux2_latent  # noqa: E402


def test_unpack_flux2_latent_shape() -> None:
    """Unpack from (1, lH*lW, 128) to (1, lH*2, lW*2, 32)."""
    latent_h = 32
    latent_w = 32
    packed = mx.zeros((1, latent_h * latent_w, 128))
    unpacked = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)
    assert unpacked.shape == (1, latent_h * 2, latent_w * 2, 32)


def test_unpack_flux2_latent_rejects_wrong_channel_count() -> None:
    packed = mx.zeros((1, 16, 64))  # 64 channels, should be 128
    with pytest.raises(ValueError, match="128-channel"):
        unpack_flux2_latent(packed, latent_height=4, latent_width=4)


def test_live_preview_callback_writes_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with fake-shaped packed latent — proves the callback wires."""
    save_path = tmp_path / "preview.png"

    # Patch from_pretrained to avoid network — use the pre-baked decoder weights.
    from mlx_taef import TAEF2

    converted = Path(__file__).parent / "converted" / "taef2_decoder.safetensors"
    real_taef2 = TAEF2.from_pretrained_local(converted)
    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(lambda cls, **kw: real_taef2))

    cb = LivePreviewCallback(
        variant="taef2",
        every=1,
        save_to=save_path,
        latent_height=32,
        latent_width=32,
    )
    fake_packed = mx.random.normal((1, 32 * 32, 128))
    cb.call_in_loop(t=0, seed=0, prompt="", latents=fake_packed, config=None, time_steps=None)
    assert save_path.exists()
    assert save_path.stat().st_size > 100  # not an empty file


def test_flux2_klein_generate_image_has_no_callbacks_kwarg() -> None:
    """Doc-shape guard: ensure README/manual examples match installed mflux API."""
    import inspect

    from mflux.models.flux2 import Flux2Klein

    sig = inspect.signature(Flux2Klein.generate_image)
    assert "callbacks" not in sig.parameters, (
        "If mflux added a callbacks kwarg, update the README to use it directly."
    )


def test_unpack_with_bn_stats_differs_from_identity_bn() -> None:
    """Non-trivial BN stats should change the output values."""
    latent_h = 4
    latent_w = 4
    packed = mx.ones((1, latent_h * latent_w, 128))
    out_identity = unpack_flux2_latent(packed, latent_height=latent_h, latent_width=latent_w)

    bn_mean = mx.ones(128) * 2.0
    bn_var = mx.ones(128) * 4.0  # std = 2.0 (approx)
    out_with_bn = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=bn_mean,
        bn_var=bn_var,
    )
    assert not np.allclose(np.array(out_identity), np.array(out_with_bn))


def test_live_preview_callback_stores_passed_flux_instance() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    sentinel = object()
    cb = LivePreviewCallback(flux=sentinel, variant="taef2", save_to="/tmp/preview.png")
    assert cb.flux is sentinel


def test_resolved_bn_is_explicit_when_kwargs_passed() -> None:
    import mlx.core as mx

    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(
        variant="taef2",
        save_to="/tmp/preview.png",
        bn_mean=mx.ones(128),
        bn_var=mx.ones(128),
    )
    assert cb.resolved_bn == "explicit"


def test_resolved_bn_is_none_when_no_bn_and_no_flux() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "none"
    assert cb.flux is None


@dataclass
class _FakeBN:
    """Minimal real-shape stand-in for mflux Flux2VAE.bn.

    Not a Mock — uses actual attribute names so the auto-extraction code
    exercises real getattr() lookups."""

    running_mean: mx.array
    running_var: mx.array
    eps: float


@dataclass
class _FakeVAE:
    bn: _FakeBN


@dataclass
class _FakeFlux:
    vae: _FakeVAE


def _build_fake_flux_with_nontrivial_bn() -> _FakeFlux:
    return _FakeFlux(
        vae=_FakeVAE(
            bn=_FakeBN(
                running_mean=mx.ones(128) * 2.0,
                running_var=mx.ones(128) * 4.0,
                eps=1e-5,
            ),
        ),
    )


def test_auto_bn_resolved_auto_when_flux_has_bn_for_taef2() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(flux=flux, variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "auto"
    assert cb.bn_mean is not None
    assert cb.bn_var is not None
    assert float(mx.max(cb.bn_mean - flux.vae.bn.running_mean)) == 0.0
    assert float(mx.max(cb.bn_var - flux.vae.bn.running_var)) == 0.0


def test_auto_bn_changes_decoded_output_vs_identity_bn() -> None:
    """End-to-end behavioral test: callback constructed with flux=fake (auto path)
    produces different decoder output than callback with no flux (identity-BN path).

    Exercises the FULL chain: construct → _try_extract_bn → callback.bn_mean/var → unpack.
    """
    from mlx_taef.integrations.mflux import LivePreviewCallback, unpack_flux2_latent

    flux = _build_fake_flux_with_nontrivial_bn()
    cb_auto = LivePreviewCallback(flux=flux, variant="taef2", save_to="/tmp/preview_auto.png")
    cb_none = LivePreviewCallback(variant="taef2", save_to="/tmp/preview_none.png")

    assert cb_auto.resolved_bn == "auto"
    assert cb_none.resolved_bn == "none"

    latent_h = 4
    latent_w = 4
    packed = mx.ones((1, latent_h * latent_w, 128))

    out_auto = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=cb_auto.bn_mean,
        bn_var=cb_auto.bn_var,
    )
    out_none = unpack_flux2_latent(
        packed,
        latent_height=latent_h,
        latent_width=latent_w,
        bn_mean=cb_none.bn_mean,  # None → identity BN
        bn_var=cb_none.bn_var,  # None → identity BN
    )

    assert not np.allclose(np.array(out_auto), np.array(out_none))


def test_explicit_kwargs_win_over_auto_bn() -> None:
    """When user passes bn_mean + bn_var AND auto_bn=True, explicit wins.
    resolved_bn reports "explicit"."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    user_mean = mx.ones(128) * 99.0
    user_var = mx.ones(128) * 99.0
    cb = LivePreviewCallback(
        flux=flux,
        variant="taef2",
        save_to="/tmp/preview.png",
        bn_mean=user_mean,
        bn_var=user_var,
    )
    assert cb.resolved_bn == "explicit"
    assert float(mx.max(cb.bn_mean - user_mean)) == 0.0


def test_auto_bn_off_when_kwarg_false() -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(
        flux=flux,
        variant="taef2",
        save_to="/tmp/preview.png",
        auto_bn=False,
    )
    assert cb.resolved_bn == "none"
    assert cb.bn_mean is None
    assert cb.bn_var is None


def test_auto_bn_resolved_none_when_flux_missing_vae(caplog) -> None:
    """flux without .vae attribute: warn + fall back to identity BN."""
    import logging

    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux_without_vae = object()
    with caplog.at_level(logging.WARNING, logger="mlx_taef"):
        cb = LivePreviewCallback(
            flux=flux_without_vae,
            variant="taef2",
            save_to="/tmp/preview.png",
        )
    assert cb.resolved_bn == "none"
    assert cb.bn_mean is None
    assert any("auto_bn" in r.message for r in caplog.records)


def test_auto_bn_resolved_none_when_variant_not_taef2() -> None:
    """auto-bn is a no-op for non-taef2 variants."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()
    cb = LivePreviewCallback(flux=flux, variant="taef1", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "none"


def _in_loop_subscribers(registry: object) -> list:
    """Return the in-loop subscriber list from a mflux CallbackRegistry.

    mflux 0.17 exposes `in_loop` (the list) and `in_loop_callbacks` (a method
    returning it). Older/newer versions may flip these; handle both shapes."""
    if hasattr(registry, "in_loop") and isinstance(registry.in_loop, list):
        return list(registry.in_loop)
    attr = getattr(registry, "in_loop_callbacks", None)
    if callable(attr):
        return list(attr())
    if isinstance(attr, list):
        return list(attr)
    raise AssertionError(
        "CallbackRegistry has no recognizable in-loop subscriber list "
        "(checked .in_loop and .in_loop_callbacks)"
    )


def test_zimage_callback_writes_png_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Z-Image reuses TAEF1 weights, so this runs fully offline on the committed taef1 file."""
    from mlx_taef import ZImage

    save_path = tmp_path / "preview.png"
    converted = Path(__file__).parent / "converted" / "taef1_decoder.safetensors"
    real_zimage = ZImage.from_pretrained_local(converted)
    monkeypatch.setattr(ZImage, "from_pretrained", classmethod(lambda cls, **kw: real_zimage))

    cb = LivePreviewCallback(
        variant="zimage", every=1, save_to=save_path, latent_height=8, latent_width=8
    )
    fake_latent = mx.random.normal((16, 1, 8, 8))  # mflux Z-Image in-loop shape
    cb.call_in_loop(t=0, seed=0, prompt="", latents=fake_latent, config=None, time_steps=None)
    assert save_path.exists()
    assert save_path.stat().st_size > 100


def test_zimage_callback_rejects_packed_flux_latent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavioral routing: the zimage callback dispatches through unpack_zimage_latent, which
    rejects a packed (1, N, 64) FLUX.1 latent — proving it is NOT wired to a flux unpack."""
    from mlx_taef import ZImage

    converted = Path(__file__).parent / "converted" / "taef1_decoder.safetensors"
    real_zimage = ZImage.from_pretrained_local(converted)
    monkeypatch.setattr(ZImage, "from_pretrained", classmethod(lambda cls, **kw: real_zimage))

    cb = LivePreviewCallback(
        variant="zimage", save_to=tmp_path / "p.png", latent_height=8, latent_width=8
    )
    packed_flux = mx.zeros((1, 64, 64))  # ndim 3 -> unpack_zimage_latent must reject
    with pytest.raises(ValueError, match=r"16, 1, h, w"):
        cb.call_in_loop(t=0, seed=0, prompt="", latents=packed_flux, config=None, time_steps=None)


def test_callback_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant must be"):
        LivePreviewCallback(variant="not-a-variant", save_to="/tmp/x.png")


@dataclass
class _FakeConfig:
    """Minimal stand-in for mflux Config — exposes .height/.width like the real one."""

    height: int
    width: int


@pytest.fixture
def offline_taef2(monkeypatch: pytest.MonkeyPatch) -> object:
    """Patch TAEF2.from_pretrained to the committed decoder weights so any
    LivePreviewCallback(variant="taef2") construction stays OFFLINE (no HF download).

    Every new taef2-constructing test in this plan must request this fixture — uncached
    `from_pretrained` downloads + converts from HF (`get_or_convert`), which would violate
    the bare-pytest-is-offline gate (Finding 4)."""
    from mlx_taef import TAEF2

    converted = Path(__file__).parent / "converted" / "taef2_decoder.safetensors"
    real_taef2 = TAEF2.from_pretrained_local(converted)
    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(lambda cls, **kw: real_taef2))
    return real_taef2


def test_partial_latent_dims_rejected_height_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-or-nothing: passing exactly one dim is a footgun — reject at construction.
    The guard must raise BEFORE model load: we patch from_pretrained to BLOW UP if reached,
    so a misplaced guard reds with AssertionError instead of silently passing on a warm cache
    (a plain pytest.raises(ValueError) can't tell 'raised before load' from 'raised after')."""
    from mlx_taef import TAEF2
    from mlx_taef.integrations.mflux import LivePreviewCallback

    def _boom(cls: object, **kw: object) -> object:
        raise AssertionError("from_pretrained reached — the all-or-nothing guard is misplaced")

    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(_boom))
    with pytest.raises(ValueError, match="together"):
        LivePreviewCallback(variant="taef2", save_to="/tmp/p.png", latent_height=32)


def test_partial_latent_dims_rejected_width_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx_taef import TAEF2
    from mlx_taef.integrations.mflux import LivePreviewCallback

    def _boom(cls: object, **kw: object) -> object:
        raise AssertionError("from_pretrained reached — the all-or-nothing guard is misplaced")

    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(_boom))
    with pytest.raises(ValueError, match="together"):
        LivePreviewCallback(variant="taef2", save_to="/tmp/p.png", latent_width=32)


def test_resolve_latent_dims_explicit_wins() -> None:
    from mlx_taef.integrations.mflux import _resolve_latent_dims

    cfg = _FakeConfig(height=1024, width=1024)
    # Explicit dims are returned untouched even when a config is present.
    assert _resolve_latent_dims(7, 9, cfg, 16) == (7, 9)


def test_resolve_latent_dims_derives_from_config_square() -> None:
    from mlx_taef.integrations.mflux import _resolve_latent_dims

    cfg = _FakeConfig(height=512, width=512)
    assert _resolve_latent_dims(None, None, cfg, 16) == (32, 32)


def test_resolve_latent_dims_derives_from_config_nonsquare() -> None:
    from mlx_taef.integrations.mflux import _resolve_latent_dims

    cfg = _FakeConfig(height=768, width=512)
    assert _resolve_latent_dims(None, None, cfg, 16) == (48, 32)


def test_resolve_latent_dims_raises_without_dims_or_config() -> None:
    from mlx_taef.integrations.mflux import _resolve_latent_dims

    with pytest.raises(ValueError, match="latent_height"):
        _resolve_latent_dims(None, None, None, 16)


def test_resolve_latent_dims_against_real_mflux_config() -> None:
    """Pin the duck-type to the REAL mflux Config (not just _FakeConfig), so a future mflux
    Config API change (e.g. the 0.18 bump) reddens this rather than passing on a stale fake.
    Config construction is pure Python (no network); verified offline."""
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_taef.integrations.mflux import _resolve_latent_dims

    cfg = Config(model_config=ModelConfig.flux2_klein_base_4b(), height=64, width=128)
    # mflux forces multiples of 16; lh = 64//16 = 4, lw = 128//16 = 8 (matches cfg.image_seq_len=32).
    assert cfg.image_seq_len == 4 * 8
    assert _resolve_latent_dims(None, None, cfg, 16) == (4, 8)


def test_binding_packed_latent_downscale_values() -> None:
    from mlx_taef.kernels import KERNELS

    assert KERNELS["taef1"].integration.packed_latent_downscale == 16
    assert KERNELS["taef2"].integration.packed_latent_downscale == 16
    # Z-Image's in-loop latent is not packed — None means "skip config-derived resolution".
    assert KERNELS["zimage"].integration.packed_latent_downscale is None


def test_live_preview_auto_resolves_dims_end_to_end(
    tmp_path: Path, offline_taef2: object
) -> None:
    """latent_height=None (the new default) -> dims derived from the Config at fire time.

    STRONG oracle: the decoded image's pixel size equals the config's image size IFF the dims
    resolved correctly. For taef2, lh = h//16, the unpack doubles to lh*2, then TAEF2 upsamples
    8x, so image_height = lh*16 = config.height. A wrong divisor (or a height/width swap) yields
    a wrong-sized image. Non-square 64x128 also catches an axis swap. Deterministic latent (mx.zeros)
    so no seed is needed (the oracle is size, not pixels)."""
    from PIL import Image

    from mlx_taef.integrations.mflux import LivePreviewCallback

    save_path = tmp_path / "preview.png"
    cb = LivePreviewCallback(variant="taef2", every=1, save_to=save_path)  # no latent dims
    assert cb.latent_height is None  # auto mode
    assert cb.latent_width is None  # auto mode

    cfg = _FakeConfig(height=64, width=128)  # downscale 16 -> lh=4, lw=8 (small + fast)
    packed = mx.zeros((1, 4 * 8, 128))
    cb.call_in_loop(t=0, seed=0, prompt="", latents=packed, config=cfg, time_steps=None)
    assert save_path.exists()
    # PIL .size is (width, height); a correct resolve gives the config's image size.
    assert Image.open(save_path).size == (128, 64)


def test_zimage_callback_auto_dims_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Z-Image's binding has packed_latent_downscale=None, so the auto path must NOT call
    _resolve_latent_dims and must NOT raise on config=None — the unpack reads dims from the
    latent's own shape. Proves the new auto-resolution code doesn't break the unpacked path."""
    from mlx_taef import ZImage
    from mlx_taef.integrations.mflux import LivePreviewCallback

    converted = Path(__file__).parent / "converted" / "taef1_decoder.safetensors"
    real_zimage = ZImage.from_pretrained_local(converted)
    monkeypatch.setattr(ZImage, "from_pretrained", classmethod(lambda cls, **kw: real_zimage))

    save_path = tmp_path / "preview.png"
    cb = LivePreviewCallback(variant="zimage", every=1, save_to=save_path)  # no latent dims
    assert cb.latent_height is None
    assert cb.latent_width is None
    assert cb._packed_downscale is None  # zimage skips config-derived resolution

    fake_latent = mx.zeros((16, 1, 8, 8))  # mflux Z-Image in-loop shape
    # config=None must be safe here (no resolution happens for the unpacked path).
    cb.call_in_loop(t=0, seed=0, prompt="", latents=fake_latent, config=None, time_steps=None)
    assert save_path.exists()
    assert save_path.stat().st_size > 100


def test_auto_bn_extracts_and_stores_eps(offline_taef2: object) -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()  # _FakeBN(eps=1e-5)
    cb = LivePreviewCallback(flux=flux, variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "auto"
    assert cb.bn_eps == 1e-5


def test_bn_eps_defaults_to_1e_4_without_flux(offline_taef2: object) -> None:
    from mlx_taef.integrations.mflux import LivePreviewCallback

    cb = LivePreviewCallback(variant="taef2", save_to="/tmp/preview.png")
    assert cb.bn_eps == 1e-4


def test_auto_bn_with_bn_missing_eps_falls_back_to_default(offline_taef2: object) -> None:
    """bn exposes running_mean/var but NO `eps` attribute: resolved_bn stays 'auto' (mean+var
    present) and bn_eps falls back to the 1e-4 default. Pins the asymmetric path where eps is
    absent but mean/var are not — otherwise an untested silent fallback."""
    from dataclasses import dataclass

    from mlx_taef.integrations.mflux import LivePreviewCallback

    @dataclass
    class _BNNoEps:  # real-shape stand-in WITHOUT an eps attribute
        running_mean: mx.array
        running_var: mx.array

    @dataclass
    class _VAE:
        bn: object

    @dataclass
    class _Flux:
        vae: object

    flux = _Flux(vae=_VAE(bn=_BNNoEps(running_mean=mx.ones(128), running_var=mx.ones(128))))
    cb = LivePreviewCallback(flux=flux, variant="taef2", save_to="/tmp/preview.png")
    assert cb.resolved_bn == "auto"  # mean + var present
    assert cb.bn_eps == 1e-4  # eps absent -> documented fallback


def test_call_in_loop_forwards_bn_eps_and_auto_dims_into_unpack_context(
    tmp_path: Path, offline_taef2: object
) -> None:
    """Storage alone is not enough (Finding 3): prove call_in_loop actually builds the
    UnpackContext with the stored bn_eps AND the auto-resolved dims. A spy binding captures
    the ctx; a stub model avoids running the real decoder."""
    from mlx_taef.integrations.mflux import LivePreviewCallback

    flux = _build_fake_flux_with_nontrivial_bn()  # eps=1e-5
    cb = LivePreviewCallback(flux=flux, variant="taef2", save_to=tmp_path / "p.png")  # auto dims
    assert cb.bn_eps == 1e-5  # extracted at construction
    assert cb._packed_downscale == 16  # read from the real taef2 binding

    captured: dict[str, object] = {}

    class _SpyBinding:
        # `unpack` is a stored callable, matching the real MfluxBinding.unpack FIELD
        # (Callable[[mx.array, UnpackContext], mx.array]) — NOT a bound method — so
        # binding.unpack(latents, ctx) passes exactly two args, faithful to the real dispatch.
        def __init__(self) -> None:
            def _capture(latents: mx.array, ctx: object) -> mx.array:
                captured["bn_eps"] = ctx.bn_eps
                captured["lh"] = ctx.latent_height
                captured["lw"] = ctx.latent_width
                return mx.zeros((1, 8, 8, 32))

            self.unpack = _capture

    class _SpyKernel:
        name = "taef2"
        integration = _SpyBinding()

    class _SpyModel:
        _kernel = _SpyKernel()

        def decode_image(self, x: mx.array) -> mx.array:
            return mx.zeros((1, 8, 8, 3), dtype=mx.uint8)

    cb.model = _SpyModel()  # type: ignore[assignment]  # swap in the spy for the fire path
    cfg = _FakeConfig(height=64, width=128)  # downscale 16 -> lh=4, lw=8
    packed = mx.zeros((1, 4 * 8, 128))
    cb.call_in_loop(t=0, seed=0, prompt="", latents=packed, config=cfg, time_steps=None)

    assert captured["bn_eps"] == 1e-5  # forwarded, not just stored
    assert (captured["lh"], captured["lw"]) == (4, 8)  # auto-resolved dims reached the ctx


def test_live_preview_callback_and_mlx_teacache_coexist_on_real_registry() -> None:
    """Both wrappers must coexist on the same mflux CallbackRegistry.
    The failure mode being asserted is callback-hook collision; mocks
    would hide it, so we use the real registry."""
    pytest.importorskip("mlx_teacache")

    from mflux.callbacks.callback_registry import CallbackRegistry

    from mlx_taef.integrations.mflux import LivePreviewCallback

    registry = CallbackRegistry()
    initial = _in_loop_subscribers(registry)

    # Build the LivePreviewCallback first (no flux instance needed for registry test)
    preview = LivePreviewCallback(variant="taef2", save_to="/tmp/preview.png")
    registry.register(preview)

    # mlx_teacache exposes a Handle that registers its own GenerationContextCallback;
    # the registration happens inside apply_teacache, which requires a flux instance.
    # We can't construct a real flux instance here without heavy model loads, so we
    # test the leaner property: the registry itself accepts arbitrary callback objects
    # without complaining about preview already being registered.
    class _FakeTeaCacheCallback:
        def __init__(self) -> None:
            self.called = 0

        def call_in_loop(self, *args: object, **kwargs: object) -> None:
            self.called += 1

    teacache_cb = _FakeTeaCacheCallback()
    registry.register(teacache_cb)

    # Both subscribers should be in the in-loop dispatch list.
    after = _in_loop_subscribers(registry)
    new_subscribers = [s for s in after if s not in initial]
    assert preview in new_subscribers
    assert teacache_cb in new_subscribers
