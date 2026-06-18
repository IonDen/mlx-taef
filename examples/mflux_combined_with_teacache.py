"""mflux + mlx-teacache + TAEF2 live preview — combined-use story.

Target search query: "mflux teacache taef", "fast FLUX previews Mac",
"combined Apple Silicon FLUX speedup".

Expected output: writes `combined_step{NN}.png` frames + a
`combined_final.webp` next to this script. Also prints the TeaCache
skip count after generation so users can see when step-skipping
actually fires.

Run with:
    uv run python examples/mflux_combined_with_teacache.py

Generates the same 4-step Flux2Klein image as the live_preview
example but wraps the transformer with mlx-teacache's apply_teacache
first. The two libraries compose cleanly — neither knows about the
other. Requires mlx-teacache >= 0.6 (installed via the [showcase]
extra: `pip install 'mlx-taef[showcase]'`).
"""

from pathlib import Path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx_teacache import apply_teacache

from mlx_taef.integrations.mflux import LivePreviewCallback

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("loading Flux2Klein base 4B (quantize=4)...")
    model = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())

    print("wrapping with mlx-teacache...")
    handle = apply_teacache(model)

    callback = LivePreviewCallback(
        flux=model,
        variant="taef2",
        every=1,
        numbered_frames=True,
        save_to=OUT_DIR / "combined.png",
        # latent_height / latent_width are auto-detected from the generation config
    )
    model.callbacks.register(callback)

    print("generating: 'a red apple on a wooden table', 4 steps + TeaCache, seed=42...")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=4,
        width=512,
        height=512,
        guidance=1.0,
    )

    final_path = OUT_DIR / "combined_final.webp"
    generated.image.save(final_path, "WEBP", quality=92)

    print(
        f"wrote {len(callback.saved_paths)} preview frames + {final_path}\n"
        f"TeaCache stats: skipped={handle.stats.skipped_count}, "
        f"computed={handle.stats.computed_count}, "
        f"variant={getattr(handle, 'variant_id', 'unknown')}"
    )


if __name__ == "__main__":
    main()
