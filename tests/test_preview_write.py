"""How LivePreviewCallback writes a preview file: atomic replace, format, mode, links.

Every preview here is a real TAEF2 decode of a tiny latent through `call_in_loop`, written
under `tmp_path`. Only Pillow's encoder, or the temp file's opener, is replaced, in the tests
that need a write to fail midway.
"""

import os
import stat
from pathlib import Path

import mlx.core as mx
import pytest

pytest.importorskip("mflux")

from PIL import Image

from mlx_taef.integrations.mflux import LivePreviewCallback

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PREVIOUS = b"the previous preview, complete"


@pytest.fixture(autouse=True)
def _offline_taef2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve TAEF2 from the committed decoder weights, so nothing downloads."""
    from mlx_taef import TAEF2

    converted = Path(__file__).parent / "converted" / "taef2_decoder.safetensors"
    real_taef2 = TAEF2.from_pretrained_local(converted)
    monkeypatch.setattr(TAEF2, "from_pretrained", classmethod(lambda cls, **kw: real_taef2))


def _callback(save_to: Path, **kwargs: object) -> LivePreviewCallback:
    return LivePreviewCallback(
        variant="taef2",
        every=1,
        save_to=save_to,
        latent_height=1,
        latent_width=1,
        **kwargs,  # type: ignore[arg-type]
    )


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _emit(cb: LivePreviewCallback) -> None:
    cb.call_in_loop(0, 0, "", mx.zeros((1, 1, 128)), None, None)


def _encoder_that_fails_midway(
    monkeypatch: pytest.MonkeyPatch, target: Path, exc: BaseException
) -> list[tuple[bytes, list[str]]]:
    """Replace Pillow's save with one that writes a partial file, records what a reader of
    `target` (and of its directory) sees at that moment, then raises `exc`."""
    seen: list[tuple[bytes, list[str]]] = []

    def _save(self: Image.Image, fp: object, format: object = None, **params: object) -> None:
        if isinstance(fp, str | os.PathLike):
            with Path(fp).open("wb") as fh:
                fh.write(_PNG_MAGIC + b"partial")
        else:
            fp.write(_PNG_MAGIC + b"partial")  # type: ignore[attr-defined]
            fp.flush()  # type: ignore[attr-defined]
        seen.append((target.read_bytes(), _names(target.parent)))
        raise exc

    monkeypatch.setattr(Image.Image, "save", _save)
    return seen


def test_a_failed_encode_leaves_the_previous_preview_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: encoding straight into the target, so a reader mid-write and the file after a
    failed encode both see a truncated image; also a temp file not removed after the failure."""
    target = tmp_path / "p.png"
    target.write_bytes(_PREVIOUS)
    seen = _encoder_that_fails_midway(monkeypatch, target, OSError("encoder failed"))
    cb = _callback(target, on_error="raise")

    with pytest.raises(OSError, match="encoder failed"):
        _emit(cb)

    during_bytes, during_listing = seen[0]
    assert during_bytes == _PREVIOUS
    assert len(during_listing) == 2  # the temp sits next to the target while it is written
    assert "p.png" in during_listing
    assert target.read_bytes() == _PREVIOUS
    assert _names(tmp_path) == ["p.png"]


def test_ctrl_c_during_the_encode_leaves_the_previous_preview_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: temp cleanup under `except Exception`, which a KeyboardInterrupt skips, leaving a
    stray temp file; or writing into the target, which Ctrl-C leaves truncated."""
    target = tmp_path / "p.png"
    target.write_bytes(_PREVIOUS)
    _encoder_that_fails_midway(monkeypatch, target, KeyboardInterrupt())
    cb = _callback(target)  # the default policy must not swallow Ctrl-C either

    with pytest.raises(KeyboardInterrupt):
        _emit(cb)

    assert target.read_bytes() == _PREVIOUS
    assert _names(tmp_path) == ["p.png"]


def test_ctrl_c_right_after_the_temp_is_created_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: the temp opened before the cleanup `try`, so a Ctrl-C that lands as the open
    returns leaves a `.tmp-*` file next to the preview."""
    from mlx_taef.integrations import mflux as integration

    real_open = integration._open_new_file

    def _open_then_interrupt(path: str, flags: int) -> int:
        os.close(real_open(path, flags))  # the temp now exists on disk
        raise KeyboardInterrupt

    monkeypatch.setattr(integration, "_open_new_file", _open_then_interrupt)
    target = tmp_path / "p.png"
    target.write_bytes(_PREVIOUS)

    with pytest.raises(KeyboardInterrupt):
        _emit(_callback(target))

    assert target.read_bytes() == _PREVIOUS
    assert _names(tmp_path) == ["p.png"]


@pytest.mark.parametrize("planted", ["file", "symlink"])
def test_an_entry_already_at_the_temp_name_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: str
) -> None:
    """Catches: the failure cleanup removing whatever sits at the temp name when the exclusive
    create refuses it, which deletes a file (or link) this process never created."""
    from mlx_taef.integrations import mflux as integration

    monkeypatch.setattr(integration.secrets, "token_hex", lambda nbytes: "00ab" * 4)
    target = tmp_path / "p.png"
    target.write_bytes(_PREVIOUS)
    taken = tmp_path / ".tmp-00ab00ab00ab00ab.png"
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"not yours")
    if planted == "file":
        taken.write_bytes(b"not yours")
    else:
        taken.symlink_to(victim)

    with pytest.raises(FileExistsError):
        _emit(_callback(target, on_error="raise"))

    assert taken.is_symlink() is (planted == "symlink")
    assert taken.read_bytes() == b"not yours"
    assert victim.read_bytes() == b"not yours"
    assert target.read_bytes() == _PREVIOUS
    assert _names(tmp_path) == sorted([".tmp-00ab00ab00ab00ab.png", "p.png", "victim.bin"])


def test_a_file_exists_error_after_the_temp_is_open_still_removes_the_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: skipping the cleanup for every FileExistsError, so one raised while writing
    leaves this process's own temp file behind."""
    target = tmp_path / "p.png"
    target.write_bytes(_PREVIOUS)
    _encoder_that_fails_midway(monkeypatch, target, FileExistsError("from the encoder"))

    with pytest.raises(FileExistsError, match="from the encoder"):
        _emit(_callback(target, on_error="raise"))

    assert target.read_bytes() == _PREVIOUS
    assert _names(tmp_path) == ["p.png"]


@pytest.mark.parametrize("link_target_exists", [True, False], ids=["existing", "dangling"])
def test_a_symlink_at_the_preview_path_is_replaced_not_written_through(
    tmp_path: Path, link_target_exists: bool
) -> None:
    """Catches: opening the target path for writing, which follows a planted link and writes
    the preview into (or creates) a file elsewhere."""
    outside = tmp_path / "elsewhere" / "victim.png"
    outside.parent.mkdir()
    if link_target_exists:
        outside.write_bytes(b"not yours to overwrite")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "p.png"
    target.symlink_to(outside)

    _emit(_callback(target, on_error="raise"))

    assert not target.is_symlink()
    assert target.read_bytes().startswith(_PNG_MAGIC)
    if link_target_exists:
        assert outside.read_bytes() == b"not yours to overwrite"
    else:
        assert not outside.exists()
    assert _names(out_dir) == ["p.png"]


def test_a_directory_at_the_preview_path_raises_and_is_left_alone(tmp_path: Path) -> None:
    """Catches: a replace step that clears or descends into an existing directory, or a temp
    left behind when the final rename fails."""
    target = tmp_path / "p.png"
    target.mkdir()
    (target / "inside.txt").write_bytes(b"keep")

    with pytest.raises(IsADirectoryError):
        _emit(_callback(target, on_error="raise"))

    assert _names(target) == ["inside.txt"]
    assert (target / "inside.txt").read_bytes() == b"keep"
    assert _names(tmp_path) == ["p.png"]


@pytest.mark.parametrize(
    ("umask", "expected_mode"), [(0o022, 0o644), (0o027, 0o640)], ids=["umask022", "umask027"]
)
def test_the_preview_file_mode_follows_the_process_umask(
    tmp_path: Path, umask: int, expected_mode: int
) -> None:
    """Catches: a temp created by `tempfile.mkstemp` (mode 0600), which leaves every preview
    unreadable to anyone but the owner, unlike a plain create."""
    previous = os.umask(umask)
    try:
        _emit(_callback(tmp_path / "p.png", on_error="raise"))
    finally:
        os.umask(previous)

    assert stat.S_IMODE((tmp_path / "p.png").stat().st_mode) == expected_mode


def test_numbered_frames_land_at_their_step_names_in_a_new_directory(tmp_path: Path) -> None:
    """Catches: the temp-and-replace writing to the temp's own name or to `save_to` instead of
    `<stem>_stepNN<suffix>`, or the missing gallery directory no longer being created."""
    cb = _callback(tmp_path / "gallery" / "preview.png", numbered_frames=True, on_error="raise")

    _emit(cb)
    _emit(cb)

    names = _names(tmp_path / "gallery")
    assert names == ["preview_step00.png", "preview_step01.png"]
    assert all((tmp_path / "gallery" / n).read_bytes().startswith(_PNG_MAGIC) for n in names)


@pytest.mark.parametrize(
    ("suffix", "is_format"),
    [
        (".png", lambda b: b.startswith(_PNG_MAGIC)),
        (".webp", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP"),
        (".jpg", lambda b: b.startswith(b"\xff\xd8\xff")),
    ],
    ids=["png", "webp", "jpg"],
)
def test_the_format_still_follows_the_target_suffix(
    tmp_path: Path, suffix: str, is_format: object
) -> None:
    """Catches: forcing PNG for every preview (a temp name without the target's suffix makes
    Pillow lose the format), which writes PNG bytes into a `.webp` or `.jpg` file."""
    target = tmp_path / f"p{suffix}"
    target.write_bytes(_PREVIOUS)

    _emit(_callback(target, on_error="raise"))

    data = target.read_bytes()
    assert is_format(data)  # type: ignore[operator]
    assert _names(tmp_path) == [target.name]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("p.notanimage", "unknown file extension: .notanimage"),
        ("preview", "unknown file extension: $"),
    ],
)
def test_an_unknown_suffix_raises_as_before_and_leaves_no_temp(
    tmp_path: Path, name: str, message: str
) -> None:
    """Catches: a temp created before Pillow rejects the suffix and never removed, or a format
    guessed for a name Pillow cannot place."""
    with pytest.raises(ValueError, match=message):
        _emit(_callback(tmp_path / name, on_error="raise"))

    assert _names(tmp_path) == []


@pytest.mark.parametrize(
    ("name", "expected_level", "magic"),
    [
        ("p.png", 1, _PNG_MAGIC),
        ("p.PNG", 1, _PNG_MAGIC),
        ("p.webp", None, b"RIFF"),
        ("p.jpg", None, b"\xff\xd8\xff"),
    ],
    ids=["png", "PNG-uppercase", "webp", "jpg"],
)
def test_only_png_previews_are_saved_at_compress_level_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    expected_level: int | None,
    magic: bytes,
) -> None:
    """Catches: PNG previews left at Pillow's default level 6 (slower to encode on every frame
    measured), an uppercase `.PNG` missed by a case-sensitive check, or a PNG-only option
    passed to the WebP or JPEG encoder.

    The spy sits on Pillow's `Image.save` and passes every call through to the real encoder.
    A PNG file does not record its zlib level (the zlib header keeps only a 2-bit speed class),
    so the options handed to Pillow are the one exact observable of the level.
    """
    real_save = Image.Image.save
    calls: list[dict[str, object]] = []

    def _spy(self: Image.Image, fp: object, format: object = None, **params: object) -> None:
        calls.append({"format": format, **params})
        real_save(self, fp, format, **params)  # type: ignore[arg-type]

    monkeypatch.setattr(Image.Image, "save", _spy)

    _emit(_callback(tmp_path / name, on_error="raise"))

    assert [call.get("compress_level") for call in calls] == [expected_level]
    assert (tmp_path / name).read_bytes().startswith(magic)
