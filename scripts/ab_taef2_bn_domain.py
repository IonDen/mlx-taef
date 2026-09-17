"""A/B: which latent domain does TAEF2 expect — batch-norm inverse applied, or not?

mflux hands the live-preview callback the *normalized* FLUX.2 latent. The full Flux2VAE applies
the batch-norm inverse (`latent * sqrt(var + eps) + mean`) before decoding. This script decodes one
captured latent three ways and scores the two TAEF2 readings against the full VAE:

- `vanilla_vae`  — mflux's full Flux2VAE (the reference image).
- `bn_inverse`   — TAEF2 on the denormalized latent (the library default with `flux=model`).
- `identity`     — TAEF2 on the normalized latent as-is (no BN statistics).

One model per subprocess; every unit writes its PNG and result JSON the moment it finishes, and a
re-run skips units whose result is already `ok`. Images are PNG so no codec sits between the
decoders and the scores.

Wall clock (M1 Max, warm HF cache): `vanilla_vae` ~3-8 min (constructs Klein base 4B at 4-bit to
reach its VAE), each TAEF2 unit ~10 s, scoring ~20 s (LPIPS is skipped with a note when the
`fixtures` dependency group is not installed).

Usage:
    uv run python scripts/ab_taef2_bn_domain.py --out-dir _artifacts/ab_taef2_bn_domain
        --latent tests/fixtures/showcase_latents/flux2_klein_base_4b.safetensors
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_taef.errors import TaefError  # noqa: E402  (after sys.path tweak)

CONDITIONS = ("vanilla_vae", "bn_inverse", "identity")
_TAEF2_CONDITIONS = ("bn_inverse", "identity")
_WORKER_TIMEOUT_S = {"vanilla_vae": 1500, "bn_inverse": 300, "identity": 300}
_TAEF2_CACHE_LIMIT_BYTES = 2 * 1024**3
# A winner has to lead by more than run-to-run and fixture noise on SSIM, and must not lose on
# LPIPS. Both TAEF2 readings differ by a ~1.8x per-channel scale, so a real effect is far larger.
_SSIM_MARGIN = 0.02


def _unit_result_path(out_dir: Path, latent_stem: str, condition: str) -> Path:
    return out_dir / f"{latent_stem}.{condition}.json"


def _unit_image_path(out_dir: Path, latent_stem: str, condition: str) -> Path:
    return out_dir / f"{latent_stem}.{condition}.png"


def _pending_units(out_dir: Path, latent_stem: str) -> list[str]:
    """Conditions still to run: no result file yet, or one whose status is not `ok`."""
    pending: list[str] = []
    for condition in CONDITIONS:
        path = _unit_result_path(out_dir, latent_stem, condition)
        if not path.exists():
            pending.append(condition)
            continue
        try:
            status = json.loads(path.read_text()).get("status")
        except json.JSONDecodeError:
            status = None
        if status != "ok":
            pending.append(condition)
    return pending


def _unpack_for(
    condition: str, latent: Any, height: int, width: int, bn_mean: Any, bn_var: Any
) -> Any:
    """Unpack the packed FLUX.2 latent into the domain `condition` names."""
    from mlx_taef.integrations.mflux import unpack_flux2_latent

    if condition not in _TAEF2_CONDITIONS:
        raise TaefError(f"not a TAEF2 condition: {condition!r}")
    use_stats = condition == "bn_inverse"
    return unpack_flux2_latent(
        latent,
        latent_height=height // 16,
        latent_width=width // 16,
        bn_mean=bn_mean if use_stats else None,
        bn_var=bn_var if use_stats else None,
    )


def _verdict(scores: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    """Pick the domain the scores support, or `inconclusive`.

    SSIM is higher-is-better, LPIPS lower-is-better. A winner must lead SSIM by `_SSIM_MARGIN`
    and, when LPIPS is available for both, must not be worse on it.
    """
    a, b = scores["bn_inverse"], scores["identity"]
    ssim_a, ssim_b = a["ssim"], b["ssim"]
    if ssim_a is None or ssim_b is None:
        return {"winner": "inconclusive", "reason": "missing SSIM"}
    gap = ssim_a - ssim_b
    if abs(gap) < _SSIM_MARGIN:
        return {"winner": "inconclusive", "reason": f"SSIM gap {gap:+.4f} inside the margin"}
    leader, trailer = ("bn_inverse", "identity") if gap > 0 else ("identity", "bn_inverse")
    lp_leader, lp_trailer = scores[leader]["lpips"], scores[trailer]["lpips"]
    if lp_leader is not None and lp_trailer is not None and lp_leader > lp_trailer:
        return {
            "winner": "inconclusive",
            "reason": f"SSIM favours {leader} but LPIPS favours {trailer}",
        }
    basis = "SSIM and LPIPS" if lp_leader is not None and lp_trailer is not None else "SSIM only"
    return {"winner": leader, "reason": f"{leader} leads on {basis} (SSIM gap {abs(gap):.4f})"}


def _save_png(image_uint8_nhwc: Any, target: Path) -> None:
    import numpy as np
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(image_uint8_nhwc[0])).save(target, "PNG")


def _worker_main(args: argparse.Namespace) -> int:
    """Decode one condition in this process; write the PNG and result JSON; exit."""
    from scripts.bench_decode import _install_memory_caps, _prep_full_vae_flux2, _resolve_cap_gb
    from scripts.run_showcase import _install_live_watchdog

    condition: str = args.worker
    stem = args.latent.stem
    result_path = _unit_result_path(args.out_dir, stem, condition)
    image_path = _unit_image_path(args.out_dir, stem, condition)
    abort_path = args.out_dir / f"{stem}.{condition}.abort.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    abort_path.unlink(missing_ok=True)

    bench_condition = "vanilla_vae" if condition == "vanilla_vae" else "taef2"
    installed_cap_gb = _install_memory_caps(_resolve_cap_gb(condition=bench_condition))

    import mlx.core as mx

    if condition in _TAEF2_CONDITIONS:
        mx.set_cache_limit(_TAEF2_CACHE_LIMIT_BYTES)

    watchdog = _install_live_watchdog(
        abort_path, f"ab_{condition}", wall_budget_s=_WORKER_TIMEOUT_S[condition] - 30.0
    )
    started = time.perf_counter()
    try:
        arrays = mx.load(str(args.latent))
        latent = arrays["latent"]
        height = int(arrays["height"].item())
        width = int(arrays["width"].item())
        if condition == "vanilla_vae":
            image = _prep_full_vae_flux2(latent, height, width)()
        else:
            from mlx_taef.api import TAEF2

            bn_mean, bn_var = arrays.get("bn_mean"), arrays.get("bn_var")
            if bn_mean is None or bn_var is None:
                raise TaefError("the latent file carries no bn_mean/bn_var; cannot run the A/B")
            taef = TAEF2.from_pretrained(include_encoder=False)
            image = taef.decode_image(
                _unpack_for(condition, latent, height, width, bn_mean, bn_var)
            )
        mx.eval(image)
        _save_png(image, image_path)
        peak_gb = mx.get_peak_memory() / 1024**3
    finally:
        watchdog.stop()

    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "condition": condition,
                "latent": args.latent.name,
                "height": height,
                "width": width,
                "image": image_path.name,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "peak_memory_gb": round(peak_gb, 3),
                "installed_cap_gb": installed_cap_gb,
            },
            indent=2,
        )
    )
    return 0


def _image_stats(ref: Path, cand: Path) -> dict[str, Any]:
    """Mean absolute error and per-channel mean shift vs the reference, in 0-255 units."""
    import numpy as np
    from PIL import Image

    r = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32)
    c = np.asarray(Image.open(cand).convert("RGB"), dtype=np.float32)
    return {
        "mae_255": round(float(np.abs(r - c).mean()), 3),
        "mean_rgb_shift_255": [
            round(float(v), 3) for v in (c.mean(axis=(0, 1)) - r.mean(axis=(0, 1)))
        ],
    }


def _score(out_dir: Path, stem: str, *, with_lpips: bool) -> dict[str, Any]:
    from scripts.run_showcase import _compute_lpips, _compute_ssim

    ref = _unit_image_path(out_dir, stem, "vanilla_vae")
    scores: dict[str, dict[str, Any]] = {}
    lpips_note: str | None = None
    for condition in _TAEF2_CONDITIONS:
        cand = _unit_image_path(out_dir, stem, condition)
        entry: dict[str, Any] = {"ssim": _compute_ssim([ref], [cand])["ssim_median"], "lpips": None}
        entry.update(_image_stats(ref, cand))
        if with_lpips and lpips_note is None:
            try:
                entry["lpips"] = _compute_lpips([ref], [cand])["lpips_median"]
            except Exception as exc:  # LPIPS is optional evidence; record why it is missing.
                lpips_note = f"{type(exc).__name__}: {exc}"
        scores[condition] = entry
    if lpips_note is not None:
        for entry in scores.values():
            entry["lpips"] = None
    return {"scores": scores, "lpips_note": lpips_note, "verdict": _verdict(scores)}


def _environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    from scripts.run_showcase import _build_hardware_metadata

    def _version_or_none(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    env: dict[str, Any] = {"hardware": _build_hardware_metadata()}
    env.update({pkg: _version_or_none(pkg) for pkg in ("mlx", "mflux", "mlx-taef")})
    return env


def _run_orchestrator(args: argparse.Namespace) -> int:
    stem = args.latent.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition in _pending_units(args.out_dir, stem):
        print(f"[run] {condition}", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            condition,
            "--latent",
            str(args.latent),
            "--out-dir",
            str(args.out_dir),
        ]
        try:
            proc = subprocess.run(cmd, timeout=_WORKER_TIMEOUT_S[condition], check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -1
        if code != 0:
            _unit_result_path(args.out_dir, stem, condition).write_text(
                json.dumps({"status": "failed", "condition": condition, "returncode": code})
            )
            print(f"[stop] {condition} failed (returncode {code})", flush=True)
            return 1
        print(f"[end] {condition}", flush=True)

    report = {
        "latent": args.latent.name,
        "units": {
            c: json.loads(_unit_result_path(args.out_dir, stem, c).read_text()) for c in CONDITIONS
        },
        **_score(args.out_dir, stem, with_lpips=not args.no_lpips),
        "environment": _environment(),
    }
    report_path = args.out_dir / f"{stem}.report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"scores": report["scores"], "verdict": report["verdict"]}, indent=2))
    print(f"[report] {report_path}", flush=True)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True, help="Captured FLUX.2 latent.")
    parser.add_argument("--out-dir", type=Path, default=Path("_artifacts/ab_taef2_bn_domain"))
    parser.add_argument("--no-lpips", action="store_true", help="Score with SSIM only.")
    parser.add_argument("--worker", choices=CONDITIONS, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the orchestrator, or one worker when `--worker` is set."""
    args = _build_argparser().parse_args(argv)
    if args.worker:
        return _worker_main(args)
    return _run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
