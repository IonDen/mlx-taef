"""Live-preview a Z-Image-Turbo generation with mlx-taef.

Run: uv run python examples/zimage_live_preview.py

Writes a low-quality TAEF1 preview every denoise step while mflux generates,
plus the final mflux-generated image as ``preview_final.webp``. Z-Image reuses
TAEF1's weights, so the preview model is a few MB and decodes in milliseconds.

Z-Image-Turbo is a distilled model (``supports_guidance=False``). mflux forces
``guidance=0.0`` and uses the ``linear`` scheduler — passing any other guidance
value has no effect.
"""

from pathlib import Path

from mlx_taef.integrations.mflux import LivePreviewCallback

OUT_DIR = Path(__file__).resolve().parent

HEIGHT = 512
WIDTH = 512


def main() -> None:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage as MfluxZImage

    print("loading Z-Image-Turbo (quantize=4)...")
    model = MfluxZImage(quantize=4, model_config=ModelConfig.z_image_turbo())

    callback = LivePreviewCallback(
        variant="zimage",
        every=1,
        numbered_frames=True,
        save_to=OUT_DIR / "preview.png",
        latent_height=HEIGHT // 8,
        latent_width=WIDTH // 8,
    )

    model.callbacks.register(callback)

    print("generating: 'a red apple on a wooden table', 4 steps, seed=42...")
    print("  guidance=0.0 (Z-Image-Turbo is distilled — mflux forces this)")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=4,
        height=HEIGHT,
        width=WIDTH,
        guidance=0.0,
    )

    final_path = OUT_DIR / "preview_final.webp"
    generated.image.save(final_path, "WEBP", quality=92)

    print(f"wrote {len(callback.saved_paths)} preview frames + {final_path}")


if __name__ == "__main__":
    main()
