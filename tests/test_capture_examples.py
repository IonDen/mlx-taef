"""Plumbing + roundtrip tests for scripts/capture_examples.py.

Generation variants (flux1-dev, flux2-klein-4b, qwen-image, krea-2-turbo) touch mflux
+ network and are tested only at the boundary: `_build_flux_model` and
`mlx_taef.integrations.mflux.LivePreviewCallback` are mocked, and the real
`_run_generation` wiring (paths, callback kwargs, generate_image kwargs) is exercised
against those mocks. Pure per-variant settings resolution (`_resolve_generation_params`,
`_GENERATION_SETTINGS`) is tested directly with no mocking.

Roundtrip variants (taesd-roundtrip, taesdxl-roundtrip) need no mflux/network — they load
already-committed MLX weights (`tests/converted/taesd_*.safetensors`) via
`Taef.from_kernel`, the same offline pattern `test_kernels_zimage.py` /
`test_kernels_krea2.py` use. `_run_roundtrip` is tested end-to-end for real, only
`_load_roundtrip_model` is monkeypatched to skip the network `from_pretrained()` call.
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

CONVERTED = Path(__file__).parent / "converted"


# --- argparse ---------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "flux1-dev",
        "flux2-klein-4b",
        "qwen-image",
        "krea-2-turbo",
        "taesd-roundtrip",
        "taesdxl-roundtrip",
    ],
)
def test_argparse_accepts_known_variants(variant: str) -> None:
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", variant])
    assert args.variant == variant


def test_argparse_rejects_unknown_variant() -> None:
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "nonexistent"])


def test_argparse_default_out_dir() -> None:
    from scripts.capture_examples import _DEFAULT_OUT_DIR, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "flux1-dev"])
    assert args.out_dir == _DEFAULT_OUT_DIR
    assert _DEFAULT_OUT_DIR.name == "examples"
    assert _DEFAULT_OUT_DIR.parent.name == "_artifacts"


def test_argparse_default_height_and_width_is_512() -> None:
    """512x512 (not mflux's own 1024x1024 CLI default): matches every existing committed
    showcase panel and fits qwen-image (a 20B q4 model) inside this machine's memory caps —
    1024x1024 caused a Metal command-buffer OOM at step 1/20 (round-1 rerun finding)."""
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "flux1-dev"])
    assert args.height == 512
    assert args.width == 512


def test_argparse_height_and_width_are_overridable() -> None:
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "flux1-dev", "--height", "768", "--width", "640"])
    assert args.height == 768
    assert args.width == 640


def test_argparse_qwen_uniform_q4_defaults_to_false() -> None:
    """Mixed precision is the default for qwen-image; --qwen-uniform-q4 opts back out."""
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "qwen-image"])
    assert args.qwen_uniform_q4 is False


def test_argparse_qwen_uniform_q4_flag_sets_true() -> None:
    from scripts.capture_examples import _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--variant", "qwen-image", "--qwen-uniform-q4"])
    assert args.qwen_uniform_q4 is True


# --- per-variant settings table (pure; verified against installed mflux 0.18.1) --------


@pytest.mark.parametrize(
    ("variant", "callback_variant", "num_steps", "guidance", "auto_bn"),
    [
        # mflux/models/flux/variants/txt2img/flux.py:150 Flux1.from_name;
        # scripts/_capture_latent.py:86-96 established repo capture defaults
        # (steps=14, guidance=3.5) for flux1-dev.
        ("flux1-dev", "taef1", 14, 3.5, False),
        # mflux/models/common/config/model_config.py:113-114 ModelConfig.flux2_klein_4b()
        # (AVAILABLE_MODELS["flux2-klein-4b"] at line 371, the DISTILLED 4B Klein, not
        # flux2_klein_base_4b); mflux/cli/defaults/defaults.py:53 MODEL_INFERENCE_STEPS
        # ["flux2-klein-4b"] = 4 (native step count); mflux/models/flux2/cli/
        # flux2_generate.py:28-31 forces guidance=1.0 for distilled (non-"base") configs.
        ("flux2-klein-4b", "taef2", 4, 1.0, True),
        # mflux/models/common/config/model_config.py:143-144 ModelConfig.qwen_image();
        # mflux/models/qwen/variants/txt2img/qwen_image.py:23-41 QwenImage class;
        # mflux/cli/defaults/defaults.py:43 MODEL_INFERENCE_STEPS["qwen-image"] = 20 (the
        # mflux CLI's own default step count); defaults.py:11 GUIDANCE_SCALE = 3.5, used
        # as the CLI's guidance fallback (models/qwen/cli/qwen_image_generate.py).
        ("qwen-image", "qwen-image", 20, 3.5, False),
        # mflux/models/common/config/model_config.py:138-139 ModelConfig.krea2()
        # (AVAILABLE_MODELS["krea-2"] at line 226); mflux/models/krea2/cli/
        # krea2_generate.py:11-12 DEFAULT_STEPS=8, DEFAULT_GUIDANCE=1.0.
        ("krea-2-turbo", "krea2", 8, 1.0, False),
    ],
)
def test_generation_settings_table(
    variant: str, callback_variant: str, num_steps: int, guidance: float, auto_bn: bool
) -> None:
    from scripts.capture_examples import _GENERATION_SETTINGS

    settings = _GENERATION_SETTINGS[variant]
    assert settings.callback_variant == callback_variant
    assert settings.num_steps == num_steps
    assert settings.guidance == guidance
    assert settings.auto_bn is auto_bn


def test_resolve_generation_params_uses_table_defaults_when_no_override() -> None:
    from scripts.capture_examples import _resolve_generation_params

    params = _resolve_generation_params("krea-2-turbo", None, None)
    assert params.callback_variant == "krea2"
    assert params.num_steps == 8
    assert params.guidance == 1.0
    assert params.auto_bn is False


def test_resolve_generation_params_applies_overrides() -> None:
    from scripts.capture_examples import _resolve_generation_params

    params = _resolve_generation_params("krea-2-turbo", 3, 2.5)
    assert params.num_steps == 3
    assert params.guidance == 2.5
    # Overrides don't touch the wiring-only fields.
    assert params.callback_variant == "krea2"
    assert params.auto_bn is False


# --- _reset_variant_dir: clears a stale variant dir before any new capture -------------


def test_reset_variant_dir_clears_prior_contents(tmp_path: Path) -> None:
    from scripts.capture_examples import _reset_variant_dir

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir(parents=True)
    stale = variant_dir / "flux1-dev_final.webp"
    stale.write_bytes(b"stale 1024x1024 leftover")

    result = _reset_variant_dir(tmp_path, "flux1-dev")

    assert result == variant_dir
    assert result.is_dir()
    assert not stale.exists()
    assert list(result.iterdir()) == []


def test_reset_variant_dir_creates_missing_dir(tmp_path: Path) -> None:
    from scripts.capture_examples import _reset_variant_dir

    result = _reset_variant_dir(tmp_path, "krea-2-turbo")

    assert result == tmp_path / "krea-2-turbo"
    assert result.is_dir()


# --- _qwen_mixed_precision_predicate: pure, no mflux/network needed --------------------


class _FakeQuantizableModule:
    """Stands in for an mlx.nn module that exposes `to_quantized` (quantizable)."""

    def to_quantized(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        pass


class _FakeNonQuantizableModule:
    """Stands in for an mlx.nn module with no `to_quantized` (e.g. an activation, a container)."""


@pytest.mark.parametrize(
    "path",
    [
        "img_in",
        "img_in.weight",
        "txt_in",
        "txt_in.bias",
        "time_text_embed",
        "time_text_embed.linear",
        "proj_out",
        "norm_out",
        "norm_out.linear",
    ],
)
def test_qwen_mixed_precision_predicate_bf16_modules_return_false(path: str) -> None:
    """The paper's protected embedding/conditioning/output modules stay bf16 (predicate
    returns False, meaning "don't quantize")."""
    from scripts.capture_examples import _qwen_mixed_precision_predicate

    assert _qwen_mixed_precision_predicate(path, _FakeQuantizableModule()) is False


@pytest.mark.parametrize(
    "path",
    [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.5.mlp.fc2",
        "transformer_blocks.54.attn.to_q",
        "transformer_blocks.59.mlp.fc2",
    ],
)
def test_qwen_mixed_precision_predicate_protected_blocks_return_q8(path: str) -> None:
    """First 6 + last 6 of 60 transformer blocks get 8-bit, group size 64."""
    from scripts.capture_examples import _qwen_mixed_precision_predicate

    result = _qwen_mixed_precision_predicate(path, _FakeQuantizableModule())
    assert result == {"group_size": 64, "bits": 8}


def test_qwen_mixed_precision_predicate_middle_blocks_return_true() -> None:
    """Blocks 6-53 (the middle 48) fall through to the default 4-bit precision (True)."""
    from scripts.capture_examples import _qwen_mixed_precision_predicate

    assert (
        _qwen_mixed_precision_predicate("transformer_blocks.30.attn.to_q", _FakeQuantizableModule())
        is True
    )
    assert (
        _qwen_mixed_precision_predicate("transformer_blocks.6.mlp.fc1", _FakeQuantizableModule())
        is True
    )
    assert (
        _qwen_mixed_precision_predicate("transformer_blocks.53.mlp.fc1", _FakeQuantizableModule())
        is True
    )


def test_qwen_mixed_precision_predicate_non_quantizable_module_returns_false() -> None:
    """A module without `to_quantized` (mlx.nn.quantize's own contract) must never be
    reported as quantizable, regardless of path — matches the unmodified
    QwenWeightDefinition.quantization_predicate's own `hasattr` guard."""
    from scripts.capture_examples import _qwen_mixed_precision_predicate

    mod = _FakeNonQuantizableModule()
    assert _qwen_mixed_precision_predicate("transformer_blocks.30.attn.to_q", mod) is False
    assert _qwen_mixed_precision_predicate("transformer_blocks.0.attn.to_q", mod) is False
    assert _qwen_mixed_precision_predicate("img_in", mod) is False


def test_qwen_mixed_precision_predicate_vae_paths_fall_through_to_default() -> None:
    """mflux applies the predicate per-component (transformer AND vae) — see
    mflux/models/common/weights/loading/weight_applier.py `_quantize`. VAE paths don't match
    any bf16/protected-block rule, so they fall through to True (uniform q4), same as the
    unmodified predicate — the recipe only changes transformer precision."""
    from scripts.capture_examples import _qwen_mixed_precision_predicate

    assert _qwen_mixed_precision_predicate("encoder.conv_in", _FakeQuantizableModule()) is True
    assert _qwen_mixed_precision_predicate("decoder.up_blocks.0", _FakeQuantizableModule()) is True


# --- _build_qwen_image_model: predicate restoration around construction ----------------


def test_build_qwen_image_model_restores_predicate_even_if_construction_raises(monkeypatch) -> None:
    """The class-attribute monkeypatch on QwenWeightDefinition must be undone even when
    building the model fails partway through — a leaked override would corrupt the NEXT
    Qwen-Image construction anywhere in the process (this mirrors the paper's own
    try/finally)."""
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from scripts import capture_examples as ce

    original = QwenWeightDefinition.__dict__["quantization_predicate"]

    class _BoomOnConstruct:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("mflux.models.qwen.variants.txt2img.qwen_image.QwenImage", _BoomOnConstruct)

    with pytest.raises(RuntimeError, match="boom"):
        ce._build_qwen_image_model(uniform_q4=False)

    assert QwenWeightDefinition.__dict__["quantization_predicate"] is original


def test_build_qwen_image_model_uniform_q4_does_not_touch_predicate(monkeypatch) -> None:
    """--qwen-uniform-q4 must not install the mixed-precision override at all."""
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from scripts import capture_examples as ce

    original = QwenWeightDefinition.__dict__["quantization_predicate"]
    sentinel = object()

    monkeypatch.setattr(
        "mflux.models.qwen.variants.txt2img.qwen_image.QwenImage",
        lambda **kwargs: sentinel,
    )

    result = ce._build_qwen_image_model(uniform_q4=True)

    assert result is sentinel
    assert QwenWeightDefinition.__dict__["quantization_predicate"] is original


# --- _run_generation: real wiring, mflux + LivePreviewCallback mocked ------------------


def _write_fake_gallery(variant_dir: Path, variant: str, num_steps: int) -> list[Path]:
    """Create `num_steps` fake numbered-frame files matching
    `LivePreviewCallback._resolve_target`'s numbered-frame naming convention
    (`<stem>_step{NN}<suffix>`, `src/mlx_taef/integrations/mflux.py:377-385`) for a
    `save_to=<variant_dir>/<variant>.webp` callback — i.e. `<variant>_step00.webp`, etc.

    Callers must invoke this from inside the fake `generate_image()`, i.e. AFTER
    `_run_generation`'s `_reset_variant_dir` call — writing the frames up front would have
    them deleted by that reset before `_validate_live_artifacts` ever sees them.
    """
    variant_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(num_steps):
        p = variant_dir / f"{variant}_step{idx:02d}.webp"
        p.write_bytes(b"frame")
        paths.append(p)
    return paths


def _make_fake_flux(
    final_image: Image.Image, *, on_generate: Callable[[], None] | None = None
) -> tuple[object, list[dict[str, object]]]:
    """A minimal fake mflux model: records callback registration + generate_image kwargs.

    `on_generate`, if given, runs inside `generate_image()` — the right point to simulate the
    real `LivePreviewCallback` writing its numbered-frame gallery to disk, since it must run
    AFTER `_run_generation`'s `_reset_variant_dir` call, not before.
    """

    class _FakeGenerated:
        image = final_image

    class _FakeCallbacks:
        def __init__(self) -> None:
            self.registered: list[object] = []

        def register(self, cb: object) -> None:
            self.registered.append(cb)

    generate_calls: list[dict[str, object]] = []

    class _FakeFlux:
        def __init__(self) -> None:
            self.callbacks = _FakeCallbacks()

        def generate_image(self, **kwargs: object) -> _FakeGenerated:
            generate_calls.append(kwargs)
            if on_generate is not None:
                on_generate()
            return _FakeGenerated()

    return _FakeFlux(), generate_calls


def test_run_generation_wires_callback_and_saves_final(tmp_path: Path, monkeypatch) -> None:
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "flux1-dev"
    final_image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    fake_flux, generate_calls = _make_fake_flux(
        final_image, on_generate=lambda: _write_fake_gallery(variant_dir, "flux1-dev", 2)
    )
    frame_paths = [variant_dir / f"flux1-dev_step{idx:02d}.webp" for idx in range(2)]
    fake_callback_instance = MagicMock()
    fake_callback_instance.saved_paths = frame_paths
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        [
            "--variant",
            "flux1-dev",
            "--out-dir",
            str(tmp_path),
            "--prompt",
            "p",
            "--seed",
            "7",
            "--num-steps",
            "2",
        ]
    )

    final_path = ce._run_generation("flux1-dev", args)

    assert final_path == variant_dir / "flux1-dev_final.webp"
    assert final_path.exists()
    assert all(p.exists() for p in frame_paths)

    fake_callback_cls.assert_called_once()
    _, kwargs = fake_callback_cls.call_args
    assert kwargs["flux"] is None  # flux1-dev has auto_bn=False
    assert kwargs["variant"] == "taef1"
    assert kwargs["every"] == 1
    assert kwargs["numbered_frames"] is True
    assert kwargs["save_to"] == variant_dir / "flux1-dev.webp"
    assert kwargs["on_error"] == "raise"

    assert fake_flux.callbacks.registered == [fake_callback_instance]
    assert len(generate_calls) == 1
    call = generate_calls[0]
    assert call["seed"] == 7
    assert call["prompt"] == "p"
    assert call["num_inference_steps"] == 2
    assert call["guidance"] == 3.5
    assert call["height"] == args.height
    assert call["width"] == args.width


def test_run_generation_passes_flux_instance_for_auto_bn_variant(
    tmp_path: Path, monkeypatch
) -> None:
    """flux2-klein-4b has auto_bn=True: LivePreviewCallback must get flux=<the model>."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "flux2-klein-4b"
    final_image = Image.new("RGB", (2, 2))
    fake_flux, _ = _make_fake_flux(
        final_image, on_generate=lambda: _write_fake_gallery(variant_dir, "flux2-klein-4b", 1)
    )
    fake_callback_instance = MagicMock()
    fake_callback_instance.saved_paths = [variant_dir / "flux2-klein-4b_step00.webp"]
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux2-klein-4b", "--out-dir", str(tmp_path), "--num-steps", "1"]
    )

    ce._run_generation("flux2-klein-4b", args)

    _, kwargs = fake_callback_cls.call_args
    assert kwargs["flux"] is fake_flux
    assert kwargs["variant"] == "taef2"
    assert kwargs["numbered_frames"] is True


def test_run_generation_raises_when_preview_gallery_incomplete(tmp_path: Path, monkeypatch) -> None:
    """A short gallery (fewer frames than num_steps) must raise, not ship silently — this is
    the regression guard for Finding 1 (single-frame mode silently dropped every step but the
    last)."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration
    from mlx_taef.errors import TaefError

    variant_dir = tmp_path / "flux1-dev"
    final_image = Image.new("RGB", (2, 2))
    # Only 1 frame written, but num_steps below asks for 2 — an incomplete gallery.
    fake_flux, _ = _make_fake_flux(
        final_image, on_generate=lambda: _write_fake_gallery(variant_dir, "flux1-dev", 1)
    )
    fake_callback_instance = MagicMock()
    fake_callback_instance.saved_paths = [variant_dir / "flux1-dev_step00.webp"]
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux1-dev", "--out-dir", str(tmp_path), "--num-steps", "2"]
    )

    with pytest.raises(TaefError, match="preview frames"):
        ce._run_generation("flux1-dev", args)


def test_run_generation_clears_stale_files_from_prior_run(tmp_path: Path, monkeypatch) -> None:
    """A rerun at a different resolution/step count must not leave old frames/final images
    mixed in with the new ones (Finding 3: a prior 1024x1024, 14-step run's leftover frames
    must not survive a 512x512, 1-step rerun). Uses a leftover frame index (`_step05`) the new
    1-step run would never itself recreate, so its continued presence can only mean the reset
    didn't happen — recreating the SAME filename (e.g. `_final.webp`) would prove nothing,
    since a working reset also legitimately produces that exact name."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir(parents=True)
    stale = variant_dir / "flux1-dev_step05.webp"  # a prior, longer run's 6th frame
    stale.write_bytes(b"stale 1024x1024 leftover")

    final_image = Image.new("RGB", (2, 2))
    fake_flux, _ = _make_fake_flux(
        final_image, on_generate=lambda: _write_fake_gallery(variant_dir, "flux1-dev", 1)
    )
    fake_callback_instance = MagicMock()
    fake_callback_instance.saved_paths = [variant_dir / "flux1-dev_step00.webp"]
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux1-dev", "--out-dir", str(tmp_path), "--num-steps", "1"]
    )

    ce._run_generation("flux1-dev", args)

    assert not stale.exists()


def test_run_generation_threads_qwen_uniform_q4_flag_to_build_flux_model(
    tmp_path: Path, monkeypatch
) -> None:
    """--qwen-uniform-q4 must reach `_build_flux_model` so the qwen-image branch can choose
    between the mixed-precision default and the plain uniform-q4 opt-out."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "qwen-image"
    final_image = Image.new("RGB", (2, 2))
    fake_flux, _ = _make_fake_flux(
        final_image, on_generate=lambda: _write_fake_gallery(variant_dir, "qwen-image", 1)
    )
    fake_callback_instance = MagicMock()
    fake_callback_instance.saved_paths = [variant_dir / "qwen-image_step00.webp"]
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    build_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_build(variant: str, **kwargs: object) -> object:
        build_calls.append((variant, kwargs))
        return fake_flux

    monkeypatch.setattr(ce, "_build_flux_model", _fake_build)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        [
            "--variant",
            "qwen-image",
            "--out-dir",
            str(tmp_path),
            "--num-steps",
            "1",
            "--qwen-uniform-q4",
        ]
    )

    ce._run_generation("qwen-image", args)

    assert build_calls == [("qwen-image", {"qwen_uniform_q4": True})]


# --- _run_roundtrip: real end-to-end against committed tiny weights --------------------


def test_run_roundtrip_end_to_end_with_tiny_weights(tmp_path: Path, monkeypatch) -> None:
    from scripts import capture_examples as ce

    from mlx_taef.api import Taef
    from mlx_taef.kernels import get_kernel

    fake_model = Taef.from_kernel(
        get_kernel("taesd"),
        decoder_path=CONVERTED / "taesd_decoder.safetensors",
        encoder_path=CONVERTED / "taesd_encoder.safetensors",
    )
    monkeypatch.setattr(ce, "_load_roundtrip_model", lambda variant: fake_model)

    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(input_path)

    out_dir = tmp_path / "out"
    parser = ce._build_argparser()
    args = parser.parse_args(
        [
            "--variant",
            "taesd-roundtrip",
            "--out-dir",
            str(out_dir),
            "--input",
            str(input_path),
        ]
    )

    input_out, roundtrip_out = ce._run_roundtrip("taesd-roundtrip", args)

    variant_dir = out_dir / "taesd-roundtrip"
    assert input_out == variant_dir / "taesd-roundtrip_input.webp"
    assert roundtrip_out == variant_dir / "taesd-roundtrip_roundtrip.webp"
    assert input_out.exists()
    assert roundtrip_out.exists()

    with Image.open(roundtrip_out) as img:
        assert img.size == (32, 32)
        assert img.mode == "RGB"
    with Image.open(input_out) as img:
        assert img.size == (32, 32)


def test_run_roundtrip_raises_without_input(tmp_path: Path) -> None:
    from scripts import capture_examples as ce

    parser = ce._build_argparser()
    args = parser.parse_args(["--variant", "taesd-roundtrip", "--out-dir", str(tmp_path)])

    with pytest.raises(ValueError, match="--input is required"):
        ce._run_roundtrip("taesd-roundtrip", args)


def test_run_roundtrip_raises_when_input_path_does_not_exist(tmp_path: Path) -> None:
    from scripts import capture_examples as ce

    from mlx_taef.errors import CaptureInputImageMissingError

    missing = tmp_path / "nope.png"
    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "taesd-roundtrip", "--out-dir", str(tmp_path), "--input", str(missing)]
    )

    with pytest.raises(CaptureInputImageMissingError, match="does not exist"):
        ce._run_roundtrip("taesd-roundtrip", args)


def test_run_roundtrip_clears_stale_files_from_prior_run(tmp_path: Path, monkeypatch) -> None:
    """Same Finding-3 guard as generation variants: a prior run's leftover output must not
    survive into a fresh roundtrip run's variant dir."""
    from scripts import capture_examples as ce

    from mlx_taef.api import Taef
    from mlx_taef.kernels import get_kernel

    fake_model = Taef.from_kernel(
        get_kernel("taesd"),
        decoder_path=CONVERTED / "taesd_decoder.safetensors",
        encoder_path=CONVERTED / "taesd_encoder.safetensors",
    )
    monkeypatch.setattr(ce, "_load_roundtrip_model", lambda variant: fake_model)

    out_dir = tmp_path / "out"
    variant_dir = out_dir / "taesd-roundtrip"
    variant_dir.mkdir(parents=True)
    stale = variant_dir / "leftover_from_a_prior_run.webp"
    stale.write_bytes(b"stale")

    input_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(input_path)

    parser = ce._build_argparser()
    args = parser.parse_args(
        [
            "--variant",
            "taesd-roundtrip",
            "--out-dir",
            str(out_dir),
            "--input",
            str(input_path),
        ]
    )

    ce._run_roundtrip("taesd-roundtrip", args)

    assert not stale.exists()


# --- main(): dispatch + guardrail wiring -----------------------------------------------


def test_main_dispatches_generation_variant_to_run_generation(tmp_path: Path, monkeypatch) -> None:
    from scripts import capture_examples as ce

    called: dict[str, object] = {}

    def _fake_run_generation(variant: str, args: object) -> Path:
        called["variant"] = variant
        final = tmp_path / "flux1-dev" / "flux1-dev_final.webp"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"")
        return final

    monkeypatch.setattr(ce, "_install_memory_caps", lambda: None)
    monkeypatch.setattr(ce, "_run_generation", _fake_run_generation)

    exit_code = ce.main(["--variant", "flux1-dev", "--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert called["variant"] == "flux1-dev"


def test_main_dispatches_roundtrip_variant_to_run_roundtrip(tmp_path: Path, monkeypatch) -> None:
    from scripts import capture_examples as ce

    called: dict[str, object] = {}

    def _fake_run_roundtrip(variant: str, args: object) -> tuple[Path, Path]:
        called["variant"] = variant
        return (tmp_path / "in.webp", tmp_path / "out.webp")

    monkeypatch.setattr(ce, "_install_memory_caps", lambda: None)
    monkeypatch.setattr(ce, "_run_roundtrip", _fake_run_roundtrip)

    exit_code = ce.main(["--variant", "taesd-roundtrip", "--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert called["variant"] == "taesd-roundtrip"


def test_main_skips_watchdog_for_roundtrip_variant(tmp_path: Path, monkeypatch) -> None:
    """The heavy-run watchdog is generation-only; a roundtrip run must never install it."""
    from scripts import capture_examples as ce

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("watchdog must not be installed for a roundtrip variant")

    monkeypatch.setattr(ce, "_install_memory_caps", lambda: None)
    monkeypatch.setattr(ce, "_install_capture_watchdog", _boom)
    monkeypatch.setattr(
        ce, "_run_roundtrip", lambda variant, args: (tmp_path / "a.webp", tmp_path / "b.webp")
    )

    exit_code = ce.main(["--variant", "taesd-roundtrip", "--out-dir", str(tmp_path)])

    assert exit_code == 0


def test_main_clears_stale_abort_artifact_before_generation(tmp_path: Path, monkeypatch) -> None:
    from scripts import capture_examples as ce

    tmp_path.mkdir(parents=True, exist_ok=True)
    stale = tmp_path / "flux1_dev.abort.json"
    stale.write_text("{}")

    def _fake_run_generation(variant: str, args: object) -> Path:
        final = tmp_path / "flux1-dev" / "flux1-dev_final.webp"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"")
        return final

    monkeypatch.setattr(ce, "_install_memory_caps", lambda: None)
    monkeypatch.setattr(ce, "_run_generation", _fake_run_generation)

    exit_code = ce.main(["--variant", "flux1-dev", "--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert not stale.exists()


# --- subprocess smoke: the exact invocation the controller uses -----------------------


class TestCliSubprocessInvocation:
    """Invokes the script the way the controller does: `python scripts/capture_examples.py`
    from the repo root, as a real subprocess — not through pytest's own sys.path (pytest's
    rootdir conftest-based collection puts the repo root on sys.path automatically, which
    hid the real bug: `sys.path[0]` for a directly-invoked script is the SCRIPT's directory
    (scripts/), not the repo root, so `from scripts._capture_latent import ...` raised
    ModuleNotFoundError before argparse ever ran). `--help` exits before any heavy work if
    reached at all; this test's job is to prove the module even loads that far.
    """

    def test_help_exits_zero_with_usage_text(self) -> None:
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/capture_examples.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "usage:" in result.stdout
        assert "--variant" in result.stdout

    def test_unknown_variant_exits_nonzero_via_argparse(self) -> None:
        """Same subprocess path, a bad --variant: argparse's own usage error, not a crash
        from the sys.path bug (which would instead surface as ModuleNotFoundError with a
        traceback, before argparse gets a chance to reject the bad value)."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/capture_examples.py", "--variant", "nonexistent"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode != 0
        assert "ModuleNotFoundError" not in result.stderr
        assert "invalid choice" in result.stderr
