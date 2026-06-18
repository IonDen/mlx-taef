"""mflux live preview with TAEF2 — headline integration.

Target search query: "mflux live preview", "FLUX2 preview mlx",
"FLUX live preview Mac".

Expected output: writes preview frames as `preview_step{NN}.png`
next to this script, plus the Flux2VAE-decoded final image as
`preview_final.png`. Per-step previews use the fast TAEF2 decoder;
the final image uses the full-quality Flux2VAE decoder.

Run with:
    uv run python examples/mflux_live_preview.py

Generates a 4-step FLUX.2 Klein base 4B image at 512x512 with a
TAEF2 live-preview callback. Demonstrates the auto-bn extraction
flow added in v0.2.0: pass `flux=model` and the callback walks
`model.vae.bn.running_mean / running_var` automatically.
"""

import time
from pathlib import Path

import mlx.core as mx
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from mlx_taef.integrations.mflux import LivePreviewCallback

OUT_DIR = Path(__file__).resolve().parent


class _TimedPreviewCallback(LivePreviewCallback):
    """LivePreviewCallback subclass that prints TAEF2 decode time per step."""

    def call_in_loop(
        self,
        t: object,
        seed: object,
        prompt: object,
        latents: mx.array,
        config: object,
        time_steps: object,
    ) -> None:
        t0 = time.perf_counter()
        super().call_in_loop(t, seed, prompt, latents, config, time_steps)
        # saved_paths is appended by the parent; its length tells us the step index.
        elapsed_ms = (time.perf_counter() - t0) * 1000
        step = len(self.saved_paths) - 1
        print(f"  {step:02d} TAEF2  {elapsed_ms:.0f}ms")


class _VaeTimer:
    """AfterLoopCallback that records a timestamp just before the Flux2VAE decode runs."""

    _t0: float | None = None

    def call_after_loop(
        self, seed: object, prompt: object, latents: mx.array, config: object
    ) -> None:
        # mflux calls this immediately before its own VAE decode; capture the start time.
        self._t0 = time.perf_counter()


def main() -> None:
    print("loading Flux2Klein base 4B (quantize=4)...")

    # Downloads black-forest-labs/FLUX.2-klein-base-4B weights and quantizes to 4-bit
    # locally. To load a pre-quantized HF repo instead, uncomment model_path= and
    # remove quantize= (e.g. "Runpod/FLUX.2-klein-4B-mflux-4bit").
    model = Flux2Klein(
        quantize=4,
        model_config=ModelConfig.flux2_klein_base_4b(),
    )

    # LivePreviewCallback auto-extracts VAE BN stats from model.vae.bn so that the
    # TAEF2 decoder produces color-correct previews without manual BN configuration.
    callback = _TimedPreviewCallback(
        flux=model,
        variant="taef2",
        every=1,
        numbered_frames=True,
        save_to=OUT_DIR / "preview.png",
        # latent_height / latent_width are auto-detected from the generation config
    )

    print(f"  resolved_bn = {callback.resolved_bn}  (expect 'auto')")
    assert callback.resolved_bn == "auto", "auto-bn extraction failed; check flux.vae.bn"

    vae_timer = _VaeTimer()
    model.callbacks.register(callback)
    model.callbacks.register(vae_timer)

    print("generating: 'a red apple on a wooden table', 4 steps, seed=42...")
    # generate_image runs the denoising loop (TAEF2 previews fire via callback) then
    # decodes the final latent with the full Flux2VAE and returns a GeneratedImage.
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=4,
        width=512,
        height=512,
        guidance=1.0,
    )

    # _VaeTimer._t0 was set just before the Flux2VAE decode; measure elapsed now.
    assert vae_timer._t0 is not None, "AfterLoopCallback did not fire"
    vae_ms = (time.perf_counter() - vae_timer._t0) * 1000
    print(f"  Final VAE  {vae_ms:.0f}ms")

    # Save the Flux2VAE-decoded result as the final output.
    final_path = OUT_DIR / "preview_final.png"
    generated.image.save(final_path, "PNG")

    print(f"wrote {len(callback.saved_paths)} preview frames + {final_path}")


if __name__ == "__main__":
    main()
