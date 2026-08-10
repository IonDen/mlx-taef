r"""Capture quality-first per-model example images for EXAMPLES.md.

`scripts/_capture_latent.py`'s showcase fixtures use `flux2_klein_base_4b` at 4 steps — a
timing protocol tuned for benchmark reproducibility, not visual quality; at 4 steps that
model produces under-denoised blobs. This script captures separate, honest example imagery
per model into `_artifacts/examples/<variant>/`, using each model's OWN native / documented
default settings rather than the showcase's fixed-step recipe.

Two kinds of variant:

- Generation variants (flux1-dev, flux2-klein-4b, qwen-image, krea-2-turbo): build the mflux
  model (quantize=4), register a `LivePreviewCallback` in NUMBERED-FRAME gallery mode
  (`every=1`, `numbered_frames=True`, `save_to=<out-dir>/<variant>/<variant>.webp`) so the
  full step progression survives, not just the last frame. Per
  `LivePreviewCallback._resolve_target` (src/mlx_taef/integrations/mflux.py:377-385) and the
  numbered-frame mode doc at :160-165/:234, that produces
  `<out-dir>/<variant>/<variant>_step00.webp ... <variant>_stepNN.webp` — one file per
  denoise step, the same mechanism `scripts/run_showcase.py`'s `_live_generation` uses
  (`save_to=save_dir / f"{scenario_dir}.webp"` + `numbered_frames=True`). The shared subject
  is the prompt "a red apple on a wooden table" at seed 42 (mirrors
  scripts/run_showcase.py's showcase prompt/seed, so example imagery and showcase imagery
  depict the same scene). The final, full-VAE-decoded image is saved to
  `<out-dir>/<variant>/<variant>_final.webp` at quality 92
  (`generated.image.save(path, "WEBP", quality=92)`), mirroring `scripts/run_showcase.py`'s
  `_live_generation`. The gallery is validated complete (one frame per step, every frame and
  the final image non-empty) via `scripts.run_showcase._validate_live_artifacts`, imported
  directly rather than reimplemented — a short/empty gallery raises `TaefError` instead of
  shipping silently incomplete output.

- Roundtrip variants (taesd-roundtrip, taesdxl-roundtrip): no mflux, no network at the model
  level beyond `from_pretrained`. Loads `--input`, encodes with `TAESD`/`TAESDXL`
  (`include_encoder=True`), decodes the result, and saves both the (RGB-normalized) input and
  the round-tripped decode to `<out-dir>/<variant>/`. `--input` dimensions should be multiples
  of 8 (the VAE's spatial downsample factor); the API itself does not enforce this (see
  `TAESD.encode`/`decode` docstrings in `src/mlx_taef/api.py`).

Per-variant generation settings, verified against the INSTALLED mflux 0.18.1 source
(`.venv/lib/python3.13/site-packages/mflux`) and cited in `_GENERATION_SETTINGS` below:

- flux1-dev: `Flux1.from_name("dev", quantize=4)`
  (mflux/models/flux/variants/txt2img/flux.py:150). 14 steps / guidance 3.5 — the repo's
  established flux1 capture default (scripts/_capture_latent.py:86-96), not the mflux CLI's
  own "dev" default of 25 steps.
- flux2-klein-4b: `Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())` — the
  DISTILLED 4B Klein (mflux/models/common/config/model_config.py:113-114, staticmethod;
  AVAILABLE_MODELS["flux2-klein-4b"] at line 371), NOT `flux2_klein_base_4b` (the showcase's
  base/timing variant). 4 steps is its native step count
  (mflux/cli/defaults/defaults.py:53, MODEL_INFERENCE_STEPS["flux2-klein-4b"] = 4). Guidance
  is forced to 1.0 for distilled (non-"base") FLUX.2 configs
  (mflux/models/flux2/cli/flux2_generate.py:28-31: "--guidance is only supported for FLUX.2
  base models").
- qwen-image: `QwenImage(quantize=4, model_config=ModelConfig.qwen_image())`
  (mflux/models/common/config/model_config.py:143-144, staticmethod; class at
  mflux/models/qwen/variants/txt2img/qwen_image.py:23-41). 20 steps is the mflux CLI's own
  default (mflux/cli/defaults/defaults.py:43, MODEL_INFERENCE_STEPS["qwen-image"] = 20;
  `QwenImage.generate_image`'s own Python default of 4 is NOT what the CLI actually uses).
  Guidance 3.5 is the CLI's guidance fallback (defaults.py:11, GUIDANCE_SCALE = 3.5, applied
  by mflux/models/qwen/cli/qwen_image_generate.py when --guidance is unset).
- krea-2-turbo: `Krea2(quantize=4, model_config=ModelConfig.krea2())`
  (mflux/models/common/config/model_config.py:138-139, staticmethod; AVAILABLE_MODELS
  ["krea-2"] at line 226). 8 steps / guidance 1.0
  (mflux/models/krea2/cli/krea2_generate.py:11-12, DEFAULT_STEPS / DEFAULT_GUIDANCE).

Resolution defaults to 512x512 — NOT mflux's own 1024x1024 CLI default
(mflux/cli/defaults/defaults.py:14, HEIGHT, WIDTH = 1024, 1024). 512x512 matches every
existing committed showcase panel (e.g. the Z-Image live-preview gallery already in
EXAMPLES.md), keeps committed webp weight small, and — unlike 1024x1024 — fits qwen-image (a
20B q4 model) inside this machine's memory caps: a 1024x1024 run died at step 1/20 with a
Metal command-buffer OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), and would have been
brutally slow (~45 s/step observed) even had it fit. `--height`/`--width` stay overridable
for anyone who wants the higher-resolution (and heavier) 1024x1024 imagery.

Heavy-run guardrails (generation variants only; roundtrip variants use tiny committed
weights and need none of this): wired-limit + soft memory caps are installed before any
model load (`_install_memory_caps`, delegating to `mlx_taef._memory_caps`), and a capture
watchdog (`scripts._capture_latent._install_capture_watchdog`, imported directly rather than
re-implemented — same discipline as `scripts/run_showcase.py`'s `_install_live_watchdog`)
aborts the run — writing `<out-dir>/<variant>.abort.json` and exiting nonzero via
`os._exit(70)` — if active memory nears the device ceiling or the wall budget is exceeded. A
stale abort artifact from a prior aborted run is cleared before each new attempt.

Publishing is atomic (mirrors the converted-weights cache's temp-then-rename discipline, see
`mlx_taef.download.get_or_convert`): both `_run_generation` and `_run_roundtrip` capture into a
same-filesystem temp sibling directory (`_temp_variant_dir`, `<out-dir>/.<variant>.tmp-<pid>`)
and only swap it into `<out-dir>/<variant>/` (`_publish_variant_dir`) after the run fully
succeeds and, for generation, the gallery validates complete. `LivePreviewCallback.save_to`
points at the temp dir during capture, so the swap is the only step that touches the published
path. On any failure the temp dir is discarded (`_discard_temp_variant_dir`) and the existing
`<out-dir>/<variant>/` — including a previously-good capture from an earlier run — is left
completely untouched; a rerun at a different resolution/step count still fully replaces the old
contents once it succeeds, just without a window where the published directory is missing or
half-written.

Wall-clock ETAs (M1 Max, quantize=4, 512x512 — documented estimates, not independently
measured at this resolution): flux1-dev ~1-2 min; flux2-klein-4b (4 native steps) under a
minute; qwen-image (20 steps) ~3-6 min; krea-2-turbo (8 steps) ~1-3 min warm cache (see
scripts/_capture_latent.py's docstring for krea-2-turbo's cold-cache download time — ~25-45
min the first time its ~36 GB of weights aren't cached). Roundtrip variants are offline and
fast (<5 s).

Qwen-Image build: mixed precision by default. A uniform `quantize=4` build of Qwen-Image shows a
documented reticulated/cracked skin-texture artifact — see the published case study,
"Qwen-Image Mixed Precision on a 32 GB Mac"
(https://ineshin.space/papers/qwen-image-mixed-precision-on-a-32-gb-mac/, recipe in its
section 4). The recipe keeps a small set of transformer modules at bf16, the first/last 6 of
60 transformer blocks at 8-bit (group size 64), and the middle 48 blocks at the requested
4-bit — see `_qwen_mixed_precision_predicate` and `_build_qwen_image_model` below, verified
against the INSTALLED mflux 0.18.1 source (not the mlx-teacache repo the paper was written
in): `mflux/models/qwen/weights/qwen_weight_definition.py`
(`QwenWeightDefinition.quantization_predicate`, same class-attribute location and
`(path, module) -> bool` signature the paper cites at mflux 0.17.5) and
`mflux/models/qwen/model/qwen_transformer/qwen_transformer.py` (60 `transformer_blocks`,
module names `img_in`/`txt_in`/`time_text_embed`/`proj_out`/`norm_out`). Pass
`--qwen-uniform-q4` for the plain uniform-q4 build instead. The paper reports about 1.9 GiB of
extra peak MLX allocation for the mixed build (27.6 -> 29.5 GiB at 512x512, 50-step CFG); this
script's 20-step, no-CFG run should sit lower, but if the active-memory watchdog aborts a
mixed-precision capture, that is an honest signal to report, not a reason to raise the memory
ceiling.

Usage:
    uv run python scripts/capture_examples.py --variant flux1-dev
    uv run python scripts/capture_examples.py --variant flux2-klein-4b
    uv run python scripts/capture_examples.py --variant qwen-image
    uv run python scripts/capture_examples.py --variant qwen-image --qwen-uniform-q4
    uv run python scripts/capture_examples.py --variant krea-2-turbo
    uv run python scripts/capture_examples.py --variant taesd-roundtrip \\
        --input path/to/photo.png
    uv run python scripts/capture_examples.py --variant taesdxl-roundtrip \\
        --input path/to/photo.png
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python scripts/capture_examples.py` (the controller's invocation) —
# adds the repo root to sys.path so `scripts._capture_latent` resolves identically to the
# test-suite path (`from scripts._capture_latent import ...`). Mirrors
# scripts/run_showcase.py's bootstrap verbatim (same mechanism, not a new one): without it,
# `sys.path[0]` is scripts/ itself when the interpreter is invoked with a script path, not
# the repo root, so `from scripts...` raises ModuleNotFoundError.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_taef.api import TAESD, TAESDXL, Taef  # noqa: E402
from mlx_taef.errors import CaptureInputImageMissingError  # noqa: E402
from scripts._capture_latent import (  # noqa: E402
    _abort_artifact_path,
    _install_capture_watchdog,
    _install_memory_caps,
)
from scripts.run_showcase import _validate_live_artifacts  # noqa: E402

_GENERATION_VARIANTS = ("flux1-dev", "flux2-klein-4b", "qwen-image", "krea-2-turbo")
_ROUNDTRIP_VARIANTS = ("taesd-roundtrip", "taesdxl-roundtrip")
_SUPPORTED_VARIANTS = [*_GENERATION_VARIANTS, *_ROUNDTRIP_VARIANTS]

_DEFAULT_OUT_DIR = Path(__file__).parent.parent / "_artifacts" / "examples"
_DEFAULT_PROMPT = "a red apple on a wooden table"

# Sized like scripts/_capture_latent.py's krea-2-turbo budget (its worst case: cold-cache
# download of ~36 GB at an observed ~25 MB/s, plus model load + generation).
_WALL_BUDGET_S = 3600.0

_WEBP_QUALITY = 92


@dataclass(frozen=True, slots=True, kw_only=True)
class _GenerationSettings:
    """Per-variant native generation recipe. See the module docstring for citations."""

    callback_variant: str
    num_steps: int
    guidance: float
    auto_bn: bool


_GENERATION_SETTINGS: dict[str, _GenerationSettings] = {
    "flux1-dev": _GenerationSettings(
        callback_variant="taef1", num_steps=14, guidance=3.5, auto_bn=False
    ),
    "flux2-klein-4b": _GenerationSettings(
        callback_variant="taef2", num_steps=4, guidance=1.0, auto_bn=True
    ),
    "qwen-image": _GenerationSettings(
        callback_variant="qwen-image", num_steps=20, guidance=3.5, auto_bn=False
    ),
    "krea-2-turbo": _GenerationSettings(
        callback_variant="krea2", num_steps=8, guidance=1.0, auto_bn=False
    ),
}

_ROUNDTRIP_CLASSES: dict[str, type[Taef]] = {
    "taesd-roundtrip": TAESD,
    "taesdxl-roundtrip": TAESDXL,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResolvedGenerationParams:
    """`_GenerationSettings` with CLI `--num-steps`/`--guidance` overrides applied."""

    callback_variant: str
    num_steps: int
    guidance: float
    auto_bn: bool


def _resolve_generation_params(
    variant: str, num_steps_override: int | None, guidance_override: float | None
) -> _ResolvedGenerationParams:
    """Merge `variant`'s table settings with any CLI overrides. Pure; no I/O."""
    settings = _GENERATION_SETTINGS[variant]
    return _ResolvedGenerationParams(
        callback_variant=settings.callback_variant,
        num_steps=num_steps_override if num_steps_override is not None else settings.num_steps,
        guidance=guidance_override if guidance_override is not None else settings.guidance,
        auto_bn=settings.auto_bn,
    )


def _temp_variant_dir(out_dir: Path, variant: str) -> Path:
    """Create and return a fresh temp sibling dir for `variant`'s in-progress capture.

    Path is `<out_dir>/.<variant>.tmp-<pid>`, a plain subdirectory of `out_dir`, so publishing
    it (`_publish_variant_dir`) is a same-filesystem rename — atomic on POSIX, never a
    half-written directory tree visible at the published path.

    Also clears any `.{variant}.tmp-*` dirs already in `out_dir` (this process's own leftover
    from an earlier failed attempt, or another pid's orphan left behind by a watchdog
    `os._exit` — see the module docstring's heavy-run guardrails section) so temp dirs never
    silently accumulate.
    """
    for stale in out_dir.glob(f".{variant}.tmp-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    tmp_dir = out_dir / f".{variant}.tmp-{os.getpid()}"
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def _publish_variant_dir(out_dir: Path, variant: str, tmp_dir: Path) -> Path:
    """Atomically swap a finished `tmp_dir` (from `_temp_variant_dir`) into `<out_dir>/<variant>/`.

    Only call this after a capture has fully succeeded (and, for generation, passed gallery
    validation) — it replaces whatever was previously published. Renames any existing
    `<out_dir>/<variant>/` aside first, renames `tmp_dir` into its place, then removes the old
    one — so a rerun fully replaces prior contents (no stale frame from a longer previous
    schedule survives) without ever deleting the published path before the replacement is
    ready to take its place.
    """
    variant_dir = out_dir / variant
    stale_dir = out_dir / f".{variant}.stale-{os.getpid()}"
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
    if variant_dir.exists():
        variant_dir.rename(stale_dir)
    tmp_dir.rename(variant_dir)
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
    return variant_dir


def _discard_temp_variant_dir(tmp_dir: Path) -> None:
    """Remove an in-progress capture's temp dir after a failed run.

    Called from a failure path only — the existing published `<out_dir>/<variant>/` is never
    touched here, which is the whole point: a failed rerun must leave a previously-good
    capture exactly as it was.
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=_SUPPORTED_VARIANTS,
        help="Model/example to capture.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory to write '<variant>/' output into. Default: %(default)s",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input image path. Required for roundtrip variants "
            "(taesd-roundtrip, taesdxl-roundtrip); ignored otherwise."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="Fixed prompt for generation variants. Default: %(default)r",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for generation variants.")
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "Image height for generation variants (matches the committed showcase's "
            "resolution; keeps qwen-image within this machine's memory caps). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "Image width for generation variants (matches the committed showcase's "
            "resolution; keeps qwen-image within this machine's memory caps). "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override inference steps for generation variants. Default: per-variant native step count.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Override CFG guidance for generation variants. Default: per-variant native guidance.",
    )
    parser.add_argument(
        "--qwen-uniform-q4",
        action="store_true",
        help=(
            "qwen-image only: build with plain uniform quantize=4 instead of the default "
            "mixed-precision recipe that avoids a documented skin-texture artifact. Ignored "
            "for every other variant."
        ),
    )
    return parser


# Qwen-Image mixed-precision recipe. See the module docstring for the full citation; verified
# against the installed mflux 0.18.1 source (mflux/models/qwen/model/qwen_transformer/
# qwen_transformer.py: 60 `transformer_blocks`, module names below).
_QWEN_MIXED_PRECISION_PROTECTED_BLOCKS = set(range(6)) | set(
    range(54, 60)
)  # first 6 + last 6 of 60
_QWEN_MIXED_PRECISION_BF16_MODULES = ("img_in", "txt_in", "time_text_embed", "proj_out", "norm_out")


def _qwen_mixed_precision_predicate(path: str, module: object) -> bool | dict[str, int]:
    """`mlx.nn.quantize`'s `class_predicate` for the Qwen-Image mixed-precision recipe.

    From "Qwen-Image Mixed Precision on a 32 GB Mac"
    (https://ineshin.space/papers/qwen-image-mixed-precision-on-a-32-gb-mac/, recipe in
    section 4): bf16 for the protected embedding/conditioning/output modules, 8-bit
    (group size 64) for the first and last 6 of the transformer's 60 blocks, and the requested
    4-bit default everywhere else quantizable.

    Applied per-component by mflux's own weight-application code
    (`mflux/models/common/weights/loading/weight_applier.py`'s `_quantize`, which calls
    `nn.quantize(model, class_predicate=weight_definition.quantization_predicate, bits=bits)`
    once per component model), so `path` here is relative to that component's own root —
    e.g. `transformer_blocks.5.attn.to_q` for the transformer, never
    `transformer.transformer_blocks.5.attn.to_q`. The predicate also runs against the VAE
    component; no VAE path matches a bf16/protected-block rule, so it falls through to the
    same `True` (uniform q4) the unmodified predicate would give it — the recipe only
    reassigns transformer precision.
    """
    if not hasattr(module, "to_quantized"):
        return False
    if any(path == p or path.startswith(p + ".") for p in _QWEN_MIXED_PRECISION_BF16_MODULES):
        return False
    if path.startswith("transformer_blocks."):
        idx = int(path.split(".")[1])
        if idx in _QWEN_MIXED_PRECISION_PROTECTED_BLOCKS:
            return {"group_size": 64, "bits": 8}
    return True


def _build_qwen_image_model(*, uniform_q4: bool) -> object:
    """Construct the Qwen-Image mflux model, quantize=4.

    Defaults to the mixed-precision recipe (`_qwen_mixed_precision_predicate`) that avoids a
    documented uniform-q4 skin-texture artifact; pass `uniform_q4=True` for the plain build
    instead. See the module docstring for the recipe's citation and memory-peak note.

    Mechanism: mflux 0.18.1 exposes the quantization predicate only as a class attribute on
    `QwenWeightDefinition` (verified: no per-instance constructor hook), so the mixed build
    monkeypatches it immediately before constructing `QwenImage` and restores the original in
    `finally` — exactly the paper's own snippet. The override is process-global for the
    duration of construction; do not construct a second Qwen-Image model concurrently on
    another thread while this runs.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage as MfluxQwenImage

    if uniform_q4:
        print("qwen-image build: uniform quantize=4 (--qwen-uniform-q4)")
        return MfluxQwenImage(quantize=4, model_config=ModelConfig.qwen_image())

    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    print(
        "qwen-image build: mixed precision (bf16 embeddings/conditioning/output, "
        "q8 group64 first+last 6 of 60 transformer blocks, q4 middle 48)"
    )
    original_predicate = QwenWeightDefinition.__dict__["quantization_predicate"]
    QwenWeightDefinition.quantization_predicate = staticmethod(_qwen_mixed_precision_predicate)
    try:
        return MfluxQwenImage(quantize=4, model_config=ModelConfig.qwen_image())
    finally:
        QwenWeightDefinition.quantization_predicate = original_predicate


def _build_flux_model(variant: str, *, qwen_uniform_q4: bool = False) -> object:
    """Construct the mflux model for a generation `variant`, quantize=4.

    Heavy: imports mflux and downloads/converts/loads weights on first call per model. See
    the module docstring for the exact `ModelConfig` constructor cited per variant.
    `qwen_uniform_q4` is read only for `variant == "qwen-image"`; see
    `_build_qwen_image_model`.
    """
    if variant == "flux1-dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        return Flux1.from_name("dev", quantize=4)
    if variant == "flux2-klein-4b":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        return Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    if variant == "qwen-image":
        return _build_qwen_image_model(uniform_q4=qwen_uniform_q4)
    if variant == "krea-2-turbo":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2

        return Krea2(quantize=4, model_config=ModelConfig.krea2())
    raise ValueError(f"unsupported generation variant: {variant}")  # pragma: no cover


def _run_generation(variant: str, args: argparse.Namespace) -> Path:
    """Generate one quality-first example image for `variant`; return the final image path.

    Captures into a temp sibling dir (`_temp_variant_dir`) and only publishes it
    (`_publish_variant_dir`) into `<out-dir>/<variant>/` after generation succeeds AND the
    gallery validates complete — see the module docstring's "Publishing is atomic" section. A
    numbered-frame `LivePreviewCallback` (`every=1`, `numbered_frames=True`) is registered at
    `<temp-dir>/<variant>.webp`, which — per `LivePreviewCallback._resolve_target` — writes one
    file per step, `<temp-dir>/<variant>_step00.webp ... <variant>_stepNN.webp`; the
    full-VAE-decoded final image is saved to `<temp-dir>/<variant>_final.webp`. On any failure
    the temp dir is discarded and the existing published dir, if any, is untouched. See the
    module docstring for the full per-variant recipe and its citations.
    """
    from mlx_taef.integrations.mflux import LivePreviewCallback

    params = _resolve_generation_params(variant, args.num_steps, args.guidance)
    tmp_dir = _temp_variant_dir(args.out_dir, variant)
    try:
        flux = _build_flux_model(variant, qwen_uniform_q4=args.qwen_uniform_q4)
        callback = LivePreviewCallback(
            flux=flux if params.auto_bn else None,
            variant=params.callback_variant,  # type: ignore[arg-type]
            every=1,
            numbered_frames=True,
            save_to=tmp_dir / f"{variant}.webp",
            on_error="raise",
        )
        flux.callbacks.register(callback)  # type: ignore[attr-defined]

        generated = flux.generate_image(  # type: ignore[attr-defined]
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=params.num_steps,
            height=args.height,
            width=args.width,
            guidance=params.guidance,
        )

        tmp_final_path = tmp_dir / f"{variant}_final.webp"
        generated.image.save(tmp_final_path, "WEBP", quality=_WEBP_QUALITY)  # type: ignore[attr-defined]
        _validate_live_artifacts(
            callback.saved_paths,
            tmp_final_path,
            expected_count=params.num_steps,  # type: ignore[attr-defined]
        )
    except BaseException:
        _discard_temp_variant_dir(tmp_dir)
        raise

    variant_dir = _publish_variant_dir(args.out_dir, variant, tmp_dir)
    return variant_dir / f"{variant}_final.webp"


def _load_roundtrip_model(variant: str) -> Taef:
    """Download/convert/load the TAESD-family model for a roundtrip `variant`.

    Heavy: network + conversion on first call, cached thereafter (`Taef.from_pretrained`).
    Kept as its own function so tests can substitute an offline `Taef.from_kernel(...)`
    instance loaded from the committed `tests/converted/` fixtures.
    """
    return _ROUNDTRIP_CLASSES[variant].from_pretrained(include_encoder=True)


def _run_roundtrip(variant: str, args: argparse.Namespace) -> tuple[Path, Path]:
    """Encode `--input` then decode it back; save both images. Return (input_path, roundtrip_path).

    Tensor layout/value-space per `src/mlx_taef/api.py`'s module docstring: `encode()` takes
    NHWC float `[0, 1]`; `decode_image()` returns NHWC uint8. `--input` is loaded via PIL,
    normalized to RGB, and given a batch axis to match that contract. Captures into a temp
    sibling dir and only publishes it into `<out-dir>/<variant>/` once both images are
    written — see `_run_generation`'s docstring and the module docstring's "Publishing is
    atomic" section for the shared mechanism.
    """
    if args.input is None:
        raise ValueError(
            f"--input is required for roundtrip variant {variant!r} (TAESD/TAESDXL "
            "roundtrip needs a source image to encode)."
        )
    if not args.input.exists():
        raise CaptureInputImageMissingError(f"--input path {args.input} does not exist.")

    import mlx.core as mx
    import numpy as np
    from PIL import Image

    tmp_dir = _temp_variant_dir(args.out_dir, variant)
    try:
        model = _load_roundtrip_model(variant)

        with Image.open(args.input) as raw:
            img = raw.convert("RGB")
        # HWC uint8 -> NHWC float32 [0, 1], the encode() contract (api.py module docstring).
        arr = np.asarray(img, dtype=np.float32) / 255.0
        batched = mx.array(arr)[None]

        latent = model.encode(batched)
        decoded = model.decode_image(latent)  # NHWC uint8, per decode_image()'s docstring.
        mx.eval(decoded)

        tmp_input_path = tmp_dir / f"{variant}_input.webp"
        tmp_roundtrip_path = tmp_dir / f"{variant}_roundtrip.webp"
        img.save(tmp_input_path, "WEBP", quality=_WEBP_QUALITY)
        Image.fromarray(np.array(decoded[0])).save(
            tmp_roundtrip_path, "WEBP", quality=_WEBP_QUALITY
        )
    except BaseException:
        _discard_temp_variant_dir(tmp_dir)
        raise

    variant_dir = _publish_variant_dir(args.out_dir, variant, tmp_dir)
    return variant_dir / f"{variant}_input.webp", variant_dir / f"{variant}_roundtrip.webp"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _install_memory_caps()

    if args.variant in _GENERATION_SETTINGS:
        # Drop any stale abort artifact from a prior aborted run before installing the
        # watchdog, so a later successful capture never leaves a misleading abort record.
        _abort_artifact_path(args.variant, args.out_dir).unlink(missing_ok=True)
        watchdog = _install_capture_watchdog(
            args.variant, args.out_dir, wall_budget_s=_WALL_BUDGET_S
        )
        try:
            final_path = _run_generation(args.variant, args)
        finally:
            watchdog.stop()
        print(f"Wrote {final_path}")
        return 0

    input_path, roundtrip_path = _run_roundtrip(args.variant, args)
    print(f"Wrote {input_path}")
    print(f"Wrote {roundtrip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
