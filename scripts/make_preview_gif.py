r"""Assemble a live-preview run's numbered frames into a side-by-side README GIF.

Pure Pillow image assembly — no MLX/mflux import, no model load. Consumes the
frame-output convention of `mlx_taef.integrations.mflux.LivePreviewCallback`
run with `numbered_frames=True`: each denoise step is written to
`<stem>_step{NN}.<ext>` next to the configured `save_to` path
(`LivePreviewCallback._resolve_target`, `src/mlx_taef/integrations/mflux.py`).
`examples/mflux_live_preview.py` uses this convention directly: it registers
`_TimedPreviewCallback(save_to=OUT_DIR / "preview.png", numbered_frames=True, ...)`,
which writes `preview_step{NN}.png` frames next to the example script, then
separately saves the full Flux2VAE-decoded result to `preview_final.png` via
`generated.image.save(...)` (outside the LivePreviewCallback gallery).

`--frames-dir` should point at the directory holding the numbered `_step{NN}`
gallery; `--final` is the separate finished-image path. Frame order comes from
the numeric `_step{NN}` suffix parsed out of each filename, not directory
listing order, so `step9` sorts before `step10`.

Usage (after running examples/mflux_live_preview.py from the repo root, which
writes its frames into the `examples/` directory). Pass `--panel-width` to
downscale each panel (LANCZOS, aspect-preserved) and hit the README's <= 5 MB
budget:
    uv run python scripts/make_preview_gif.py \\
        --frames-dir examples \\
        --frames-glob "preview_step*.png" \\
        --final examples/preview_final.png \\
        --out docs/assets/live-preview.gif \\
        --panel-width 384
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from mlx_taef.errors import PreviewFramesMissingError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "docs" / "assets" / "live-preview.gif"

# Matches the LivePreviewCallback numbered-frame convention (`<stem>_step{NN}<ext>`)
# while excluding a separately-saved final image such as `preview_final.png` or
# `<scenario>_final.webp` (run_showcase.py's convention), neither of which contains
# the literal "_step" substring.
_DEFAULT_FRAMES_GLOB = "*_step*.*"
_STEP_RE = re.compile(r"_step(\d+)")

_CAPTION_HEIGHT_PX = 24
_CAPTION_BG = (255, 255, 255)
_CAPTION_FG = (0, 0, 0)
_TARGET_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_FPS = 10.0


def _frame_step(path: Path) -> int:
    """Parse the `_step{NN}` index out of a preview frame filename.

    Raises ValueError with a clear message if the filename doesn't carry the
    `_step{NN}` convention `LivePreviewCallback` writes in numbered-frame mode.
    """
    match = _STEP_RE.search(path.stem)
    if match is None:
        raise ValueError(
            f"frame {path.name!r} does not carry a '_step<N>' suffix (the "
            "LivePreviewCallback numbered-frame convention); check --frames-glob "
            "or the source of these frames."
        )
    return int(match.group(1))


def _discover_frames(frames_dir: Path, frames_glob: str) -> list[Path]:
    """Return preview frame paths sorted by their numeric `_step{NN}` index.

    Raises PreviewFramesMissingError if `frames_glob` matches nothing in
    `frames_dir` — an empty gallery would otherwise silently produce a
    zero-length (invalid) GIF.
    """
    matches = list(frames_dir.glob(frames_glob))
    if not matches:
        raise PreviewFramesMissingError(
            f"no frames matched glob {frames_glob!r} in {frames_dir} — did the "
            "live-preview run finish, or does --frames-glob need adjusting?"
        )
    return sorted(matches, key=lambda p: (_frame_step(p), p.name))


def _load_rgb(path: Path) -> Image.Image:
    """Open an image file and normalize to RGB (drops alpha/palette variance)."""
    with Image.open(path) as img:
        return img.convert("RGB")


def _resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    """Resize `img` to `target_width`, preserving aspect ratio (LANCZOS).

    A no-op when `img` is already `target_width` wide (avoids a redundant
    resample pass, which keeps output byte-identical across runs).
    """
    if img.width == target_width:
        return img
    ratio = target_width / img.width
    target_height = max(1, round(img.height * ratio))
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _compose_frame(
    preview: Image.Image, final: Image.Image, *, step: int, max_step: int
) -> Image.Image:
    """Paste `preview` (left) and `final` (right) onto one canvas with a step caption."""
    width = preview.width + final.width
    height = max(preview.height, final.height) + _CAPTION_HEIGHT_PX
    canvas = Image.new("RGB", (width, height), _CAPTION_BG)
    canvas.paste(preview, (0, 0))
    canvas.paste(final, (preview.width, 0))

    draw = ImageDraw.Draw(canvas)
    caption = f"step {step:02d} / {max_step:02d}  |  preview vs. final decode"
    left, top, right, bottom = draw.textbbox((0, 0), caption)
    text_x = max(0, (width - (right - left)) // 2)
    band_top = max(preview.height, final.height)
    text_y = band_top + max(0, (_CAPTION_HEIGHT_PX - (bottom - top)) // 2)
    draw.text((text_x, text_y), caption, fill=_CAPTION_FG)
    return canvas


def assemble_side_by_side_gif(
    frames_dir: Path,
    final_path: Path,
    out_path: Path,
    *,
    frames_glob: str = _DEFAULT_FRAMES_GLOB,
    fps: float = _DEFAULT_FPS,
    panel_width: int | None = None,
) -> Path:
    """Assemble numbered preview frames + a final still into a side-by-side GIF.

    Deterministic given the same frame files: no timestamps, randomness, or
    filesystem-listing-order dependence — frames are ordered by their parsed
    `_step{NN}` index, and Pillow's GIF quantization/encoding is deterministic
    for identical pixel input (holds with `panel_width` set too, since LANCZOS
    resampling is itself deterministic for fixed input pixels + target size).

    Args:
        frames_dir: directory holding the `LivePreviewCallback` numbered-frame
            gallery (see module docstring for the naming convention).
        final_path: the finished, full-VAE-decoded image shown as the static
            right-hand panel.
        out_path: where to write the assembled GIF. Parent directories are
            created if missing.
        frames_glob: glob (within `frames_dir`) selecting the numbered frames.
        fps: animation rate for the left (preview) panel. Must be positive.
        panel_width: when set, each panel (preview frames + the final still)
            is downscaled to this width in pixels before compositing,
            preserving aspect ratio (LANCZOS). Default `None` keeps the
            frames' native size. The caption band's own height is unaffected
            (it sits below the panels regardless of their scaled size), so a
            smaller `panel_width` shrinks the combined GIF's width and panel
            height while the caption strip stays legible.

    Returns:
        `out_path`, for convenient chaining.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if panel_width is not None and panel_width <= 0:
        raise ValueError(f"panel_width must be positive, got {panel_width!r}")
    frame_paths = _discover_frames(frames_dir, frames_glob)
    steps = [_frame_step(p) for p in frame_paths]
    max_step = max(steps)

    final_img = _load_rgb(final_path)
    first_preview = _load_rgb(frame_paths[0])
    if panel_width is not None:
        first_preview = _resize_to_width(first_preview, panel_width)
    frame_size = first_preview.size
    final_resized = final_img.resize(frame_size, Image.Resampling.LANCZOS)

    composed = []
    for path, step in zip(frame_paths, steps, strict=True):
        preview = first_preview if path == frame_paths[0] else _load_rgb(path)
        if preview.size != frame_size:
            preview = preview.resize(frame_size, Image.Resampling.LANCZOS)
        composed.append(_compose_frame(preview, final_resized, step=step, max_step=max_step))

    duration_ms = round(1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=composed[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing the numbered '_step{NN}' preview frames.",
    )
    parser.add_argument(
        "--frames-glob",
        default=_DEFAULT_FRAMES_GLOB,
        help="Glob (within --frames-dir) selecting the numbered frames. Default: %(default)s",
    )
    parser.add_argument(
        "--final",
        type=Path,
        required=True,
        help="Path to the full-VAE-decoded final image.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output GIF path. Default: %(default)s",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=_DEFAULT_FPS,
        help="Animation frame rate for the preview side. Default: %(default)s",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=None,
        help=(
            "Downscale each panel (preview + final) to this width in pixels, "
            "preserving aspect ratio (LANCZOS), before compositing. "
            "Default: native (no resize)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    out_path = assemble_side_by_side_gif(
        args.frames_dir,
        args.final,
        args.out,
        frames_glob=args.frames_glob,
        fps=args.fps,
        panel_width=args.panel_width,
    )
    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    if size_bytes > _TARGET_MAX_BYTES:
        print(
            f"warning: {out_path} is {size_mb:.2f} MB, over the "
            f"{_TARGET_MAX_BYTES / 1024 / 1024:.0f} MB README target — consider fewer "
            "frames, a lower --fps, or downscaling the source frames.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
