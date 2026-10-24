"""The bfloat16-vs-float32 preview shift reported by scripts/measure_preview_dtype_shift.py.

Runs the script's CLI entry for each committed latent, offline, on the committed tiny decoders.
A bfloat16 decode of these latents differs from the float32 decode by a few 255ths, so a max of
zero means the script compared a decode with itself.
"""

import json

import pytest


def _run(case: str, capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    from scripts.measure_preview_dtype_shift import main

    assert main(["--case", case]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1, lines
    result: dict[str, object] = json.loads(lines[0])
    return result


@pytest.mark.parametrize(
    ("case", "latent"),
    [
        ("zimage", "tests/fixtures/showcase_latents/z_image_turbo.safetensors"),
        ("flux2-klein", "tests/fixtures/showcase_latents/flux2_klein_base_4b.safetensors"),
    ],
)
def test_the_bfloat16_preview_differs_from_the_float32_preview_by_a_few_255ths(
    case: str, latent: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The quoted shift is a few 255ths (Z-Image max 6, Klein max 2). Catches: the "before"
    decode getting a float32 cast back anywhere (callback, unpack or decoder input), which
    compares float32 with float32 and reports a zero max and mean; or the two images taken
    from different latents or decoders, which reports a shift far above 8/255."""
    result = _run(case, capsys)

    assert set(result) == {"case", "latent", "max_abs_255", "mean_abs_255", "mlx", "pixels"}
    assert result["case"] == case
    assert result["latent"] == latent
    max_abs = result["max_abs_255"]
    mean_abs = result["mean_abs_255"]
    assert isinstance(max_abs, int)
    assert isinstance(mean_abs, float)
    assert 1 <= max_abs <= 8, result
    assert mean_abs > 0, result


def test_an_unknown_case_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """Catches: a mistyped case falling back to a default latent and printing its numbers
    under the wrong name."""
    from scripts.measure_preview_dtype_shift import main

    with pytest.raises(SystemExit) as exc:
        main(["--case", "flux1"])

    assert exc.value.code == 2
    assert capsys.readouterr().out == ""
