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


# --- _temp_variant_dir / _publish_variant_dir / _discard_temp_variant_dir: atomic publish ---


def test_temp_variant_dir_creates_a_fresh_dir(tmp_path: Path) -> None:
    from scripts.capture_examples import _temp_variant_dir

    result = _temp_variant_dir(tmp_path, "flux1-dev")

    assert result.is_dir()
    assert result.parent == tmp_path
    assert result.name.startswith(".flux1-dev.tmp-")
    assert list(result.iterdir()) == []


def test_temp_variant_dir_clears_leftover_temp_dirs_from_other_pids(tmp_path: Path) -> None:
    """An orphaned temp dir from a killed prior process (e.g. a watchdog `os._exit`) must not
    accumulate forever — a fresh call clears any `.{variant}.tmp-*` match, not just its own pid."""
    from scripts.capture_examples import _temp_variant_dir

    orphan = tmp_path / ".flux1-dev.tmp-99999999"
    orphan.mkdir()
    (orphan / "leftover.webp").write_bytes(b"orphaned frame")

    result = _temp_variant_dir(tmp_path, "flux1-dev")

    assert not orphan.exists()
    assert result.is_dir()


def test_temp_variant_dir_does_not_touch_other_variants(tmp_path: Path) -> None:
    from scripts.capture_examples import _temp_variant_dir

    other = tmp_path / ".krea-2-turbo.tmp-1"
    other.mkdir()
    (other / "frame.webp").write_bytes(b"x")

    _temp_variant_dir(tmp_path, "flux1-dev")

    assert other.exists()


def test_publish_variant_dir_swaps_temp_into_place_and_removes_old(tmp_path: Path) -> None:
    from scripts.capture_examples import _publish_variant_dir, _temp_variant_dir

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir()
    (variant_dir / "old.webp").write_bytes(b"old capture")

    tmp_dir = _temp_variant_dir(tmp_path, "flux1-dev")
    (tmp_dir / "new.webp").write_bytes(b"new capture")

    result = _publish_variant_dir(tmp_path, "flux1-dev", tmp_dir)

    assert result == variant_dir
    assert (variant_dir / "new.webp").read_bytes() == b"new capture"
    assert not (variant_dir / "old.webp").exists()
    assert not tmp_dir.exists()


def test_publish_variant_dir_with_no_prior_published_dir(tmp_path: Path) -> None:
    """Publishing must also work the first time, when `<out_dir>/<variant>/` doesn't exist yet."""
    from scripts.capture_examples import _publish_variant_dir, _temp_variant_dir

    tmp_dir = _temp_variant_dir(tmp_path, "flux1-dev")
    (tmp_dir / "new.webp").write_bytes(b"new capture")

    result = _publish_variant_dir(tmp_path, "flux1-dev", tmp_dir)

    assert result == tmp_path / "flux1-dev"
    assert (result / "new.webp").read_bytes() == b"new capture"
    assert not tmp_dir.exists()


def test_discard_temp_variant_dir_removes_the_dir(tmp_path: Path) -> None:
    from scripts.capture_examples import _discard_temp_variant_dir

    tmp_dir = tmp_path / ".flux1-dev.tmp-1"
    tmp_dir.mkdir()
    (tmp_dir / "partial.webp").write_bytes(b"x")

    _discard_temp_variant_dir(tmp_dir)

    assert not tmp_dir.exists()


def test_discard_temp_variant_dir_is_a_noop_when_already_gone(tmp_path: Path) -> None:
    from scripts.capture_examples import _discard_temp_variant_dir

    tmp_dir = tmp_path / ".flux1-dev.tmp-999"

    _discard_temp_variant_dir(tmp_dir)  # must not raise

    assert not tmp_dir.exists()


def _spawn_and_reap_dead_pid() -> int:
    """Spawn a short-lived subprocess, wait for it to exit, and return its now-dead pid.

    More robust than guessing an arbitrary unused pid number. Uses `Popen` + `wait()` rather
    than `subprocess.run(...).pid` — `run()` returns a `CompletedProcess`, which has no `.pid`.
    """
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=5)
    return proc.pid


# --- _extract_pid_suffix / _is_pid_alive: pure pid parsing + liveness ------------------


@pytest.mark.parametrize(
    ("dirname", "expected"),
    [
        (".flux1-dev.tmp-12345", 12345),
        (".flux1-dev.stale-1", 1),
        (".krea-2-turbo.tmp-999", 999),  # variant name itself contains hyphens
    ],
)
def test_extract_pid_suffix_parses_trailing_int(dirname: str, expected: int) -> None:
    from scripts.capture_examples import _extract_pid_suffix

    assert _extract_pid_suffix(dirname) == expected


def test_extract_pid_suffix_returns_none_for_non_numeric_suffix() -> None:
    from scripts.capture_examples import _extract_pid_suffix

    assert _extract_pid_suffix(".flux1-dev.tmp-notapid") is None


def test_is_pid_alive_true_for_self() -> None:
    import os

    from scripts.capture_examples import _is_pid_alive

    assert _is_pid_alive(os.getpid()) is True


def test_is_pid_alive_false_for_a_dead_pid() -> None:
    from scripts.capture_examples import _is_pid_alive

    assert _is_pid_alive(_spawn_and_reap_dead_pid()) is False


# --- _recover_variant_dir: repairs a crash inside _publish_variant_dir's rename window --


def test_recover_variant_dir_restores_newest_stale_dir_when_published_dir_missing(
    tmp_path: Path,
) -> None:
    """RED case: simulates the exact on-disk state a kill between `_publish_variant_dir`'s two
    renames leaves — no `<variant>/`, a populated `.{variant}.stale-<dead-pid>/` — and asserts
    the next invocation's recovery restores it byte-intact."""
    from scripts.capture_examples import _recover_variant_dir

    # A guaranteed-dead pid, to stand in for "the process that died mid-publish".
    dead_pid = _spawn_and_reap_dead_pid()

    stale_dir = tmp_path / f".flux1-dev.stale-{dead_pid}"
    stale_dir.mkdir()
    good_bytes = b"a previously-good capture, orphaned mid-swap"
    (stale_dir / "flux1-dev_final.webp").write_bytes(good_bytes)

    _recover_variant_dir(tmp_path, "flux1-dev")

    variant_dir = tmp_path / "flux1-dev"
    assert variant_dir.is_dir()
    assert (variant_dir / "flux1-dev_final.webp").read_bytes() == good_bytes
    assert not stale_dir.exists()


def test_recover_variant_dir_picks_the_newest_stale_dir_when_several_exist(
    tmp_path: Path,
) -> None:
    import time

    from scripts.capture_examples import _recover_variant_dir

    older = tmp_path / f".flux1-dev.stale-{_spawn_and_reap_dead_pid()}"
    older.mkdir()
    (older / "flux1-dev_final.webp").write_bytes(b"older")

    time.sleep(0.05)  # ensure a distinguishable mtime ordering

    newer = tmp_path / f".flux1-dev.stale-{_spawn_and_reap_dead_pid()}"
    newer.mkdir()
    (newer / "flux1-dev_final.webp").write_bytes(b"newer")

    _recover_variant_dir(tmp_path, "flux1-dev")

    variant_dir = tmp_path / "flux1-dev"
    assert (variant_dir / "flux1-dev_final.webp").read_bytes() == b"newer"
    assert not older.exists()
    assert not newer.exists()


def test_recover_variant_dir_sweeps_stale_dirs_alongside_existing_published_dir(
    tmp_path: Path,
) -> None:
    """When `<variant>/` already exists (the common case — no crash happened), any leftover
    `.{variant}.stale-*` dirs must still be swept, not accumulate forever."""
    from scripts.capture_examples import _recover_variant_dir

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir()
    (variant_dir / "flux1-dev_final.webp").write_bytes(b"currently published")

    stale_dir = tmp_path / f".flux1-dev.stale-{_spawn_and_reap_dead_pid()}"
    stale_dir.mkdir()
    (stale_dir / "flux1-dev_final.webp").write_bytes(b"old, superseded")

    _recover_variant_dir(tmp_path, "flux1-dev")

    assert (variant_dir / "flux1-dev_final.webp").read_bytes() == b"currently published"
    assert not stale_dir.exists()


def test_recover_variant_dir_leaves_a_live_pids_stale_dir_untouched(tmp_path: Path) -> None:
    """The LOW-severity single-writer guard: a stale dir whose embedded pid is still alive
    (an in-flight concurrent `_publish_variant_dir` on the same variant) must not be stolen or
    swept — even though the published dir is missing."""
    import os

    from scripts.capture_examples import _recover_variant_dir

    live_stale = tmp_path / f".flux1-dev.stale-{os.getpid()}"
    live_stale.mkdir()
    (live_stale / "flux1-dev_final.webp").write_bytes(b"still being published by a live pid")

    _recover_variant_dir(tmp_path, "flux1-dev")

    assert not (tmp_path / "flux1-dev").exists()
    assert live_stale.exists()


def test_temp_variant_dir_triggers_recovery(tmp_path: Path) -> None:
    """Integration check: `_temp_variant_dir` (called at the start of every real run) performs
    the recovery pass, not just `_recover_variant_dir` in isolation."""
    from scripts.capture_examples import _temp_variant_dir

    stale_dir = tmp_path / f".flux1-dev.stale-{_spawn_and_reap_dead_pid()}"
    stale_dir.mkdir()
    good_bytes = b"orphaned by a mid-publish crash"
    (stale_dir / "flux1-dev_final.webp").write_bytes(good_bytes)

    _temp_variant_dir(tmp_path, "flux1-dev")

    variant_dir = tmp_path / "flux1-dev"
    assert (variant_dir / "flux1-dev_final.webp").read_bytes() == good_bytes
    assert not stale_dir.exists()


def test_temp_variant_dir_does_not_sweep_a_tmp_dir_whose_pid_is_alive(tmp_path: Path) -> None:
    """The LOW-severity single-writer guard applied to the `.tmp-*` sweep: a concurrent
    same-variant run's live temp dir must not be deleted out from under it. Uses a real,
    separately-running subprocess for the "live" pid — reusing our own pid would collide with
    the path `_temp_variant_dir` creates for itself."""
    import subprocess
    import sys

    from scripts.capture_examples import _temp_variant_dir

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        live_tmp = tmp_path / f".flux1-dev.tmp-{proc.pid}"
        live_tmp.mkdir()
        frame = live_tmp / "flux1-dev_step00.webp"
        frame.write_bytes(b"a live concurrent run's in-progress frame")

        result = _temp_variant_dir(tmp_path, "flux1-dev")

        assert result != live_tmp
        assert frame.exists()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


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


def _spy(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> list[object]:
    """Wrap `module.name` with a spy that still calls the original, recording each call's
    return value.

    Used to learn the (unpredictable, pid-based) temp dir a real `_temp_variant_dir` call
    produced — that path isn't known until `_run_generation`/`_run_roundtrip` actually calls
    it, so tests can't precompute it the way they could when captures wrote straight into a
    predictable `<out_dir>/<variant>/`.
    """
    original = getattr(module, name)
    results: list[object] = []

    def _wrapped(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(module, name, _wrapped)
    return results


def _write_fake_gallery(dest_dir: Path, variant: str, num_steps: int) -> list[Path]:
    """Create `num_steps` fake numbered-frame files matching
    `LivePreviewCallback._resolve_target`'s numbered-frame naming convention
    (`<stem>_step{NN}<suffix>`, `src/mlx_taef/integrations/mflux.py:377-385`) for a
    `save_to=<dest_dir>/<variant>.webp` callback — i.e. `<variant>_step00.webp`, etc.

    Callers must invoke this from inside the fake `generate_image()`, using the temp dir
    `_run_generation`'s `_temp_variant_dir` call produced (see `_spy`) — writing frames
    anywhere else means `_validate_live_artifacts` never sees them.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(num_steps):
        p = dest_dir / f"{variant}_step{idx:02d}.webp"
        p.write_bytes(b"frame")
        paths.append(p)
    return paths


def _make_fake_flux(
    final_image: Image.Image, *, on_generate: Callable[[], None] | None = None
) -> tuple[object, list[dict[str, object]]]:
    """A minimal fake mflux model: records callback registration + generate_image kwargs.

    `on_generate`, if given, runs inside `generate_image()` — the right point to simulate the
    real `LivePreviewCallback` writing its numbered-frame gallery to disk, since it must run
    AFTER `_run_generation`'s `_temp_variant_dir` call, not before.
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

    tmp_dirs = _spy(monkeypatch, ce, "_temp_variant_dir")
    variant_dir = tmp_path / "flux1-dev"
    final_image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    fake_callback_instance = MagicMock()

    def _on_generate() -> None:
        fake_callback_instance.saved_paths = _write_fake_gallery(tmp_dirs[-1], "flux1-dev", 2)

    fake_flux, generate_calls = _make_fake_flux(final_image, on_generate=_on_generate)
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

    tmp_dir = tmp_dirs[0]
    assert not tmp_dir.exists()  # published: renamed into place, not left behind
    assert final_path == variant_dir / "flux1-dev_final.webp"
    assert final_path.exists()
    assert (variant_dir / "flux1-dev_step00.webp").exists()
    assert (variant_dir / "flux1-dev_step01.webp").exists()

    fake_callback_cls.assert_called_once()
    _, kwargs = fake_callback_cls.call_args
    assert kwargs["flux"] is None  # flux1-dev has auto_bn=False
    assert kwargs["variant"] == "taef1"
    assert kwargs["every"] == 1
    assert kwargs["numbered_frames"] is True
    assert kwargs["save_to"] == tmp_dir / "flux1-dev.webp"  # temp dir during capture
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

    tmp_dirs = _spy(monkeypatch, ce, "_temp_variant_dir")
    final_image = Image.new("RGB", (2, 2))
    fake_callback_instance = MagicMock()

    def _on_generate() -> None:
        fake_callback_instance.saved_paths = _write_fake_gallery(tmp_dirs[-1], "flux2-klein-4b", 1)

    fake_flux, _ = _make_fake_flux(final_image, on_generate=_on_generate)
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
    last). (c): the temp dir must also be cleaned up, not left behind."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration
    from mlx_taef.errors import TaefError

    tmp_dirs = _spy(monkeypatch, ce, "_temp_variant_dir")
    final_image = Image.new("RGB", (2, 2))
    fake_callback_instance = MagicMock()

    def _on_generate() -> None:
        # Only 1 frame written, but num_steps below asks for 2 — an incomplete gallery.
        fake_callback_instance.saved_paths = _write_fake_gallery(tmp_dirs[-1], "flux1-dev", 1)

    fake_flux, _ = _make_fake_flux(final_image, on_generate=_on_generate)
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux1-dev", "--out-dir", str(tmp_path), "--num-steps", "2"]
    )

    with pytest.raises(TaefError, match="preview frames"):
        ce._run_generation("flux1-dev", args)

    assert not tmp_dirs[-1].exists()
    assert not (tmp_path / "flux1-dev").exists()  # never published


def test_run_generation_failure_leaves_existing_variant_dir_byte_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """(a) A failed generation must leave a pre-existing, previously-good variant dir
    completely untouched — the regression this atomic-publish scheme fixes: capturing
    straight into the published dir meant a mid-run crash destroyed a good prior capture with
    no recovery."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir(parents=True)
    good_final = variant_dir / "flux1-dev_final.webp"
    good_bytes = b"a previously-good capture that must survive a failed rerun"
    good_final.write_bytes(good_bytes)

    class _FailingCallbacks:
        def register(self, cb: object) -> None:
            pass

    class _FailingFlux:
        def __init__(self) -> None:
            self.callbacks = _FailingCallbacks()

        def generate_image(self, **kwargs: object) -> object:
            raise RuntimeError("simulated mid-run failure")

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: _FailingFlux())
    monkeypatch.setattr(
        mflux_integration, "LivePreviewCallback", MagicMock(return_value=MagicMock())
    )

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux1-dev", "--out-dir", str(tmp_path), "--num-steps", "1"]
    )

    with pytest.raises(RuntimeError, match="simulated mid-run failure"):
        ce._run_generation("flux1-dev", args)

    assert good_final.read_bytes() == good_bytes
    assert {p.name for p in variant_dir.iterdir()} == {"flux1-dev_final.webp"}
    assert list(tmp_path.glob(".flux1-dev.tmp-*")) == []  # no leftover temp dir


def test_run_generation_success_replaces_old_contents_fully(tmp_path: Path, monkeypatch) -> None:
    """(b) A successful run must fully replace old published contents — no stale frame from a
    longer previous schedule survives alongside a shorter new one."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    variant_dir = tmp_path / "flux1-dev"
    variant_dir.mkdir(parents=True)
    stale = variant_dir / "flux1-dev_step09.webp"  # a longer previous run's 10th frame
    stale.write_bytes(b"stale, from a 14-step run")

    tmp_dirs = _spy(monkeypatch, ce, "_temp_variant_dir")
    final_image = Image.new("RGB", (2, 2))
    fake_callback_instance = MagicMock()

    def _on_generate() -> None:
        fake_callback_instance.saved_paths = _write_fake_gallery(tmp_dirs[-1], "flux1-dev", 1)

    fake_flux, _ = _make_fake_flux(final_image, on_generate=_on_generate)
    fake_callback_cls = MagicMock(return_value=fake_callback_instance)

    monkeypatch.setattr(ce, "_build_flux_model", lambda variant, **kwargs: fake_flux)
    monkeypatch.setattr(mflux_integration, "LivePreviewCallback", fake_callback_cls)

    parser = ce._build_argparser()
    args = parser.parse_args(
        ["--variant", "flux1-dev", "--out-dir", str(tmp_path), "--num-steps", "1"]
    )

    ce._run_generation("flux1-dev", args)

    assert not stale.exists()
    assert {p.name for p in variant_dir.iterdir()} == {
        "flux1-dev_final.webp",
        "flux1-dev_step00.webp",
    }


def test_run_generation_threads_qwen_uniform_q4_flag_to_build_flux_model(
    tmp_path: Path, monkeypatch
) -> None:
    """--qwen-uniform-q4 must reach `_build_flux_model` so the qwen-image branch can choose
    between the mixed-precision default and the plain uniform-q4 opt-out."""
    from scripts import capture_examples as ce

    import mlx_taef.integrations.mflux as mflux_integration

    tmp_dirs = _spy(monkeypatch, ce, "_temp_variant_dir")
    final_image = Image.new("RGB", (2, 2))
    fake_callback_instance = MagicMock()

    def _on_generate() -> None:
        fake_callback_instance.saved_paths = _write_fake_gallery(tmp_dirs[-1], "qwen-image", 1)

    fake_flux, _ = _make_fake_flux(final_image, on_generate=_on_generate)
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
    assert list(out_dir.glob(".taesd-roundtrip.tmp-*")) == []  # published, no leftover temp

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


def test_run_roundtrip_failure_leaves_existing_variant_dir_byte_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """(a) A failed roundtrip must leave a pre-existing, previously-good variant dir
    completely untouched."""
    from scripts import capture_examples as ce

    out_dir = tmp_path / "out"
    variant_dir = out_dir / "taesd-roundtrip"
    variant_dir.mkdir(parents=True)
    good_file = variant_dir / "taesd-roundtrip_roundtrip.webp"
    good_bytes = b"a previously-good roundtrip capture that must survive a failed rerun"
    good_file.write_bytes(good_bytes)

    def _boom(variant: str) -> object:
        raise RuntimeError("simulated model-load failure")

    monkeypatch.setattr(ce, "_load_roundtrip_model", _boom)

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

    with pytest.raises(RuntimeError, match="simulated model-load failure"):
        ce._run_roundtrip("taesd-roundtrip", args)

    assert good_file.read_bytes() == good_bytes
    assert {p.name for p in variant_dir.iterdir()} == {"taesd-roundtrip_roundtrip.webp"}
    assert list(out_dir.glob(".taesd-roundtrip.tmp-*")) == []  # no leftover temp dir


def test_run_roundtrip_success_replaces_old_contents_fully(tmp_path: Path, monkeypatch) -> None:
    """(b) A successful roundtrip must fully replace old published contents."""
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
    assert {p.name for p in variant_dir.iterdir()} == {
        "taesd-roundtrip_input.webp",
        "taesd-roundtrip_roundtrip.webp",
    }


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
