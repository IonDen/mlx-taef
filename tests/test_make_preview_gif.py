"""Tests for scripts/make_preview_gif.py — pure Pillow assembly, no model load.

Builds tiny synthetic frames under tmp_path (never touches examples/ output or
any real preview run) to exercise frame discovery, side-by-side composition,
determinism, and the empty-frames-dir error path.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


def _make_frame(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def _make_synthetic_run(
    tmp_path: Path, *, n_frames: int = 3, frame_size: tuple[int, int] = (16, 12)
) -> tuple[Path, Path]:
    """Write `n_frames` numbered preview frames + a final still under tmp_path.

    Mirrors the LivePreviewCallback(numbered_frames=True) convention:
    `preview_step{NN}.png` frames plus a separately named final image.
    Returns (frames_dir, final_path).
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(n_frames):
        # A distinct color per step so frames are visibly different.
        color = (10 * i % 256, 20 * i % 256, 30 * i % 256)
        _make_frame(frames_dir / f"preview_step{i:02d}.png", size=frame_size, color=color)
    final_path = tmp_path / "preview_final.png"
    _make_frame(final_path, size=frame_size, color=(200, 200, 200))
    return frames_dir, final_path


def test_assemble_writes_an_animated_gif_with_one_frame_per_step(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=3)
    out_path = tmp_path / "out.gif"

    result = assemble_side_by_side_gif(frames_dir, final_path, out_path)

    assert result == out_path
    assert out_path.exists()
    with Image.open(out_path) as gif:
        assert gif.n_frames == 3


def test_assembled_gif_frame_size_matches_side_by_side_geometry(tmp_path: Path) -> None:
    from scripts.make_preview_gif import _CAPTION_HEIGHT_PX, assemble_side_by_side_gif

    frame_size = (16, 12)
    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2, frame_size=frame_size)
    out_path = tmp_path / "out.gif"

    assemble_side_by_side_gif(frames_dir, final_path, out_path)

    with Image.open(out_path) as gif:
        expected_width = frame_size[0] * 2  # preview + final, same width since final is resized
        expected_height = frame_size[1] + _CAPTION_HEIGHT_PX
        assert gif.size == (expected_width, expected_height)


def test_assemble_is_byte_identical_across_runs(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=4)
    out_path_a = tmp_path / "a.gif"
    out_path_b = tmp_path / "b.gif"

    assemble_side_by_side_gif(frames_dir, final_path, out_path_a)
    assemble_side_by_side_gif(frames_dir, final_path, out_path_b)

    assert out_path_a.read_bytes() == out_path_b.read_bytes()


def test_empty_frames_dir_raises_clear_error(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    from mlx_taef.errors import PreviewFramesMissingError

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    final_path = tmp_path / "final.png"
    _make_frame(final_path, size=(8, 8), color=(0, 0, 0))

    with pytest.raises(PreviewFramesMissingError, match="frames"):
        assemble_side_by_side_gif(frames_dir, final_path, tmp_path / "out.gif")


def test_frames_sort_numerically_not_lexicographically(tmp_path: Path) -> None:
    """step9 must sort before step10 — lexicographic sort would get this backwards."""
    from scripts.make_preview_gif import _discover_frames

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frame(frames_dir / "preview_step9.png", size=(4, 4), color=(1, 1, 1))
    _make_frame(frames_dir / "preview_step10.png", size=(4, 4), color=(2, 2, 2))
    _make_frame(frames_dir / "preview_step0.png", size=(4, 4), color=(3, 3, 3))

    frames = _discover_frames(frames_dir, "*_step*.*")

    assert [p.name for p in frames] == [
        "preview_step0.png",
        "preview_step9.png",
        "preview_step10.png",
    ]


def test_frames_glob_excludes_final_image_named_without_step_suffix(tmp_path: Path) -> None:
    """A final image saved as `<stem>_final.<ext>` (run_showcase.py's convention) must
    not be swept up by the default numbered-frame glob."""
    from scripts.make_preview_gif import _DEFAULT_FRAMES_GLOB, _discover_frames

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _make_frame(frames_dir / "scenario_step00.webp", size=(4, 4), color=(1, 1, 1))
    _make_frame(frames_dir / "scenario_final.webp", size=(4, 4), color=(2, 2, 2))

    frames = _discover_frames(frames_dir, _DEFAULT_FRAMES_GLOB)

    assert [p.name for p in frames] == ["scenario_step00.webp"]


def test_final_image_is_resized_to_match_preview_frame_size(tmp_path: Path) -> None:
    """A final image with a different native size than the preview frames must still
    produce a valid side-by-side canvas sized off the preview frame dimensions."""
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_size = (16, 12)
    _make_frame(frames_dir / "preview_step00.png", size=frame_size, color=(50, 60, 70))
    final_path = tmp_path / "preview_final.png"
    _make_frame(final_path, size=(40, 30), color=(200, 200, 200))  # different native size
    out_path = tmp_path / "out.gif"

    assemble_side_by_side_gif(frames_dir, final_path, out_path)

    with Image.open(out_path) as gif:
        assert gif.size[0] == frame_size[0] * 2


def test_fps_controls_frame_duration(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2)
    out_path = tmp_path / "out.gif"

    assemble_side_by_side_gif(frames_dir, final_path, out_path, fps=10.0)

    with Image.open(out_path) as gif:
        assert gif.info["duration"] == 100  # 1000ms / 10fps


def test_non_positive_fps_rejected(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2)

    with pytest.raises(ValueError, match="fps"):
        assemble_side_by_side_gif(frames_dir, final_path, tmp_path / "out.gif", fps=0.0)


def test_cli_main_writes_output_and_returns_zero(tmp_path: Path) -> None:
    from scripts.make_preview_gif import main

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=3)
    out_path = tmp_path / "out.gif"

    exit_code = main(
        [
            "--frames-dir",
            str(frames_dir),
            "--final",
            str(final_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    assert out_path.exists()


def test_panel_width_downscales_output_and_stays_animated(tmp_path: Path) -> None:
    """--panel-width halves each panel's width; output width is 2*N (no gutter)."""
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frame_size = (64, 48)
    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=3, frame_size=frame_size)
    out_path = tmp_path / "out.gif"
    target_width = frame_size[0] // 2  # 32

    assemble_side_by_side_gif(frames_dir, final_path, out_path, panel_width=target_width)

    with Image.open(out_path) as gif:
        assert gif.size[0] == target_width * 2
        assert gif.n_frames == 3


def test_panel_width_preserves_aspect_ratio(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frame_size = (64, 48)  # 4:3
    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2, frame_size=frame_size)
    out_path = tmp_path / "out.gif"
    target_width = 32

    assemble_side_by_side_gif(frames_dir, final_path, out_path, panel_width=target_width)

    with Image.open(out_path) as gif:
        # panel height should scale by the same ratio as width: 48 * (32/64) = 24,
        # plus the (unscaled) caption band.
        from scripts.make_preview_gif import _CAPTION_HEIGHT_PX

        assert gif.size[1] == 24 + _CAPTION_HEIGHT_PX


def test_panel_width_output_is_deterministic(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=4, frame_size=(64, 48))
    out_path_a = tmp_path / "a.gif"
    out_path_b = tmp_path / "b.gif"

    assemble_side_by_side_gif(frames_dir, final_path, out_path_a, panel_width=32)
    assemble_side_by_side_gif(frames_dir, final_path, out_path_b, panel_width=32)

    assert out_path_a.read_bytes() == out_path_b.read_bytes()


def test_panel_width_none_is_native_and_unchanged() -> None:
    """Default (no --panel-width) keeps existing native-size behavior — regression guard
    for the pre-existing geometry test."""
    import inspect

    from scripts.make_preview_gif import assemble_side_by_side_gif

    sig = inspect.signature(assemble_side_by_side_gif)
    assert sig.parameters["panel_width"].default is None


def test_non_positive_panel_width_rejected(tmp_path: Path) -> None:
    from scripts.make_preview_gif import assemble_side_by_side_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2)

    with pytest.raises(ValueError, match="panel_width"):
        assemble_side_by_side_gif(frames_dir, final_path, tmp_path / "out.gif", panel_width=0)


def test_cli_panel_width_flag_downscales_output(tmp_path: Path) -> None:
    from scripts.make_preview_gif import main

    frame_size = (64, 48)
    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2, frame_size=frame_size)
    out_path = tmp_path / "out.gif"

    exit_code = main(
        [
            "--frames-dir",
            str(frames_dir),
            "--final",
            str(final_path),
            "--out",
            str(out_path),
            "--panel-width",
            "32",
        ]
    )

    assert exit_code == 0
    with Image.open(out_path) as gif:
        assert gif.size[0] == 64  # 2 * 32


def test_documented_direct_script_invocation_produces_a_gif(tmp_path: Path) -> None:
    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=3)
    out_path = tmp_path / "out.gif"
    repo_root = Path(__file__).parent.parent

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "make_preview_gif.py"),
            "--frames-dir",
            str(frames_dir),
            "--final",
            str(final_path),
            "--out",
            str(out_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert out_path.exists()


def test_main_warns_on_stderr_when_output_exceeds_target_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real assembled GIF is nowhere near the 5 MB README target, so this drives it
    over the line by lowering `_TARGET_MAX_BYTES` (monkeypatched module constant) below
    the tiny synthetic GIF's actual size, rather than generating megabytes of frames."""
    import scripts.make_preview_gif as make_preview_gif

    frames_dir, final_path = _make_synthetic_run(tmp_path, n_frames=2)
    out_path = tmp_path / "out.gif"
    monkeypatch.setattr(make_preview_gif, "_TARGET_MAX_BYTES", 1)

    exit_code = make_preview_gif.main(
        [
            "--frames-dir",
            str(frames_dir),
            "--final",
            str(final_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert str(out_path) in captured.err
    assert "MB" in captured.err
