"""A/B: which latent domain does TAEF2 expect — batch-norm inverse applied, or not?

mflux hands the live-preview callback the *normalized* FLUX.2 latent. The full Flux2VAE applies
the batch-norm inverse (`latent * sqrt(var + eps) + mean`) before decoding. This script decodes one
captured latent three ways and scores the two TAEF2 readings against the full VAE:

- `vanilla_vae`  — mflux's full Flux2VAE (the reference image).
- `bn_inverse`   — TAEF2 on the denormalized latent (the library default up to v0.8.1).
- `identity`     — TAEF2 on the normalized latent as-is (the default since v0.8.2).

Every unit writes its PNG and result JSON the moment it finishes, and a re-run skips units whose
result is already `ok`. Images are PNG so no codec sits between the decoders and the scores.

One model per subprocess; results are keyed on the latent's sha256, so pointing the script at a
different latent never re-scores another latent's images.

Wall clock (M1 Max, warm HF cache): a few seconds per unit (mflux materializes only the VAE), plus
~20 s of scoring. LPIPS needs the `fixtures` dependency group; a missing or failing scorer is
recorded in `lpips_note` and the verdict falls back to SSIM.

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
# Every worker bounds MLX's retained-buffer pool: its default limit sits near device memory, and
# the watchdog samples active memory only.
_CACHE_LIMIT_BYTES = {
    "vanilla_vae": 6 * 1024**3,
    "bn_inverse": 2 * 1024**3,
    "identity": 2 * 1024**3,
}
# A winner has to lead by more than run-to-run and fixture noise on SSIM, and must not lose on
# LPIPS. Both TAEF2 readings differ by a ~1.8x per-channel scale, so a real effect is far larger.
_SSIM_MARGIN = 0.02


def _unit_result_path(out_dir: Path, latent_stem: str, condition: str) -> Path:
    return out_dir / f"{latent_stem}.{condition}.json"


def _unit_image_path(out_dir: Path, latent_stem: str, condition: str) -> Path:
    return out_dir / f"{latent_stem}.{condition}.png"


def _cache_limit_bytes(condition: str) -> int:
    return _CACHE_LIMIT_BYTES[condition]


def _install_worker_limits(condition: str) -> int:
    """Pin the wired/soft memory caps and bound the MLX cache pool; return the wired cap in GB."""
    import mlx.core as mx

    from scripts import bench_decode

    bench_condition = "vanilla_vae" if condition == "vanilla_vae" else "taef2"
    installed_cap_gb = bench_decode._install_memory_caps(
        bench_decode._resolve_cap_gb(condition=bench_condition)
    )
    mx.set_cache_limit(_cache_limit_bytes(condition))
    return installed_cap_gb


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pending_units(out_dir: Path, latent_stem: str, *, latent_sha256: str) -> list[str]:
    """Conditions still to run: no `ok` result yet for this exact latent."""
    pending: list[str] = []
    for condition in CONDITIONS:
        path = _unit_result_path(out_dir, latent_stem, condition)
        if not path.exists():
            pending.append(condition)
            continue
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            result = {}
        if result.get("status") != "ok" or result.get("latent_sha256") != latent_sha256:
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
    from scripts.bench_decode import _prep_full_vae_flux2
    from scripts.run_showcase import _install_live_watchdog

    condition: str = args.worker
    stem = args.latent.stem
    result_path = _unit_result_path(args.out_dir, stem, condition)
    image_path = _unit_image_path(args.out_dir, stem, condition)
    abort_path = args.out_dir / f"{stem}.{condition}.abort.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    abort_path.unlink(missing_ok=True)

    installed_cap_gb = _install_worker_limits(condition)

    import mlx.core as mx

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
                "latent_sha256": _sha256(args.latent),
                "height": height,
                "width": width,
                "image": image_path.name,
                # Whole-unit wall clock and whole-process peak (load + decode + PNG write); the
                # decode-only figures live in the showcase report.
                "unit_wall_s": round(time.perf_counter() - started, 3),
                "process_peak_memory_gb": round(peak_gb, 3),
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
    from scripts import run_showcase

    ref = _unit_image_path(out_dir, stem, "vanilla_vae")
    notes: list[str] = []
    score_fn = None
    if with_lpips:
        try:
            score_fn = run_showcase._build_lpips_score_fn()
        except Exception as exc:  # LPIPS is optional evidence; record why it is missing.
            notes.append(f"scorer unavailable: {type(exc).__name__}: {exc}")

    scores: dict[str, dict[str, Any]] = {}
    for condition in _TAEF2_CONDITIONS:
        cand = _unit_image_path(out_dir, stem, condition)
        entry: dict[str, Any] = {
            "ssim": run_showcase._compute_ssim([ref], [cand])["ssim_median"],
            "lpips": None,
        }
        entry.update(_image_stats(ref, cand))
        if score_fn is not None:
            try:
                entry["lpips"] = run_showcase._compute_lpips([ref], [cand], score_fn=score_fn)[
                    "lpips_median"
                ]
            except Exception as exc:  # one arm failing must not discard the other's score
                notes.append(f"{condition}: {type(exc).__name__}: {exc}")
        scores[condition] = entry
    return {
        "scores": scores,
        "lpips_note": "; ".join(notes) or None,
        "verdict": _verdict(scores),
    }


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


def _staging_dir(out_dir: Path, latent_sha256: str) -> Path:
    return out_dir / f".staging-{latent_sha256[:12]}"


def _is_published(out_dir: Path, stem: str, latent_sha256: str) -> bool:
    """True when `out_dir` already holds a complete, scored result for this exact latent."""
    try:
        report = json.loads((out_dir / f"{stem}.report.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return report.get("latent_sha256") == latent_sha256 and not _pending_units(
        out_dir, stem, latent_sha256=latent_sha256
    )


def _publish(staging: Path, out_dir: Path) -> None:
    for path in sorted(staging.iterdir()):
        path.replace(out_dir / path.name)
    staging.rmdir()


def _run_orchestrator(args: argparse.Namespace) -> int:
    """Run the pending units in a staging directory; publish into `--out-dir` only as a whole.

    `--out-dir` may already hold another latent's results under the same file names. Units are
    therefore written to `.staging-<sha>` and moved over in one step after every unit succeeded
    and the report was scored, so a failed or interrupted run never leaves an old report next to
    images decoded from a different latent. A later run resumes from the staged units.
    """
    stem = args.latent.stem
    latent_sha256 = _sha256(args.latent)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{stem}.report.json"
    if _is_published(args.out_dir, stem, latent_sha256):
        report = json.loads(report_path.read_text())
        print(json.dumps({"scores": report["scores"], "verdict": report["verdict"]}, indent=2))
        print(f"[skip] already measured for this latent: {report_path}", flush=True)
        return 0

    staging = _staging_dir(args.out_dir, latent_sha256)
    staging.mkdir(exist_ok=True)
    for condition in _pending_units(staging, stem, latent_sha256=latent_sha256):
        print(f"[run] {condition}", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            condition,
            "--latent",
            str(args.latent),
            "--out-dir",
            str(staging),
        ]
        try:
            proc = subprocess.run(cmd, timeout=_WORKER_TIMEOUT_S[condition], check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -1
        if code != 0:
            _unit_result_path(staging, stem, condition).write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "condition": condition,
                        "latent_sha256": latent_sha256,
                        "returncode": code,
                    }
                )
            )
            print(f"[stop] {condition} failed (returncode {code}); staged in {staging}", flush=True)
            return 1
        print(f"[end] {condition}", flush=True)

    report = {
        "latent": args.latent.name,
        "latent_sha256": latent_sha256,
        "units": {
            c: json.loads(_unit_result_path(staging, stem, c).read_text()) for c in CONDITIONS
        },
        **_score(staging, stem, with_lpips=not args.no_lpips),
        "environment": _environment(),
    }
    (staging / report_path.name).write_text(json.dumps(report, indent=2))
    _publish(staging, args.out_dir)
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
