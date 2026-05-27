"""mflux live preview with TAEF2 — headline integration.

Target search query: "mflux live preview", "FLUX2 preview mlx",
"FLUX live preview Mac".

Expected output: writes preview frames as `preview_step{NN}.png`
next to this script, plus the final mflux-generated image as
`preview_final.webp`. With auto-bn enabled (default), previews
are color-correct from step 1.

Run with:
    uv run python examples/mflux_live_preview.py

Generates a 4-step FLUX.2 Klein base 4B image at 512x512 with a
TAEF2 live-preview callback. Demonstrates the auto-bn extraction
flow added in v0.2.0: pass `flux=model` and the callback walks
`model.vae.bn.running_mean / running_var` automatically.
"""

from pathlib import Path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from mlx_taef.integrations.mflux import LivePreviewCallback

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("loading Flux2Klein base 4B (quantize=4)...")
    model = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())

    callback = LivePreviewCallback(
        flux=model,
        variant="taef2",
        every=1,
        numbered_frames=True,
        save_to=OUT_DIR / "preview.png",
        latent_height=32,
        latent_width=32,
    )
    print(f"  resolved_bn = {callback.resolved_bn}  (expect 'auto')")
    assert callback.resolved_bn == "auto", "auto-bn extraction failed; check flux.vae.bn"

    model.callbacks.register(callback)

    print("generating: 'a red apple on a wooden table', 4 steps, seed=42...")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=4,
        width=512,
        height=512,
        guidance=1.0,
    )

    final_path = OUT_DIR / "preview_final.webp"
    generated.image.save(final_path, "WEBP", quality=92)

    print(f"wrote {len(callback.saved_paths)} preview frames + {final_path}")


if __name__ == "__main__":
    main()
