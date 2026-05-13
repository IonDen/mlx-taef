"""Generate reference fixtures by running PyTorch TAESD on fixed-seed inputs.

Run once: `python tests/generate_references.py`.
Outputs are committed to tests/reference/.

Handles both key formats: upstream-Sequential (.safetensors with "0.weight"-style
keys, used by taesd/taesdxl HF repos) and Diffusers (TAEF1/TAEF2, requires
convert_diffusers_sd_to_taesd documented in TAEF2 model card).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.numpy import save_file
from safetensors.torch import load_file as safetensors_torch_load

REFERENCE_DIR = Path(__file__).parent / "reference"
SEEDS = [42, 123, 2024, 7777, 31415]

# (variant_name, hf_repo, hf_filename, key_format, latent_channels, arch_variant)
VARIANTS = [
    ("taesd", "madebyollin/taesd", "taesd_decoder.safetensors", "upstream", 4, None),
    ("taesdxl", "madebyollin/taesdxl", "taesdxl_decoder.safetensors", "upstream", 4, None),
    ("taef1", "madebyollin/taef1", "diffusion_pytorch_model.safetensors", "diffusers", 16, None),
    ("taef2", "madebyollin/taef2", "taef2.safetensors", "diffusers", 32, "flux_2"),
]


def convert_diffusers_sd_to_taesd(sd: dict, *, role: str) -> dict:
    """Apply Diffusers-key -> upstream-Sequential-key mapping.

    Per TAEF2 model card: Diffusers keys are 'encoder.layers.<i>.weight' etc.
    For the decoder, the Diffusers VAE prepends one layer that the upstream
    Sequential decoder does not have, so all decoder indices shift by +1.
    Encoder keys have no offset.
    """
    out: dict = {}
    prefix = f"{role}."
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if suffix.startswith("layers."):
            parts = suffix.split(".")
            idx = int(parts[1])
            if role == "decoder":
                idx += 1
            new_key = f"{idx}." + ".".join(parts[2:])
        else:
            new_key = suffix
        out[new_key] = v
    return out


def main() -> int:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent / "_third_party"))
    from taesd import TAESD

    for variant_name, repo, filename, key_format, latent_channels, arch_variant in VARIANTS:
        print(f"=== {variant_name} ===")
        weights_path = hf_hub_download(repo_id=repo, filename=filename)
        raw_sd = safetensors_torch_load(weights_path)

        if key_format == "diffusers":
            decoder_sd = convert_diffusers_sd_to_taesd(raw_sd, role="decoder")
            encoder_sd = convert_diffusers_sd_to_taesd(raw_sd, role="encoder")
        else:
            decoder_sd = raw_sd
            enc_filename = filename.replace("_decoder.", "_encoder.")
            enc_path = hf_hub_download(repo_id=repo, filename=enc_filename)
            encoder_sd = safetensors_torch_load(enc_path)

        model = TAESD(
            encoder_path=None,
            decoder_path=None,
            latent_channels=latent_channels,
            arch_variant=arch_variant,
        ).eval()
        model.decoder.load_state_dict(decoder_sd)
        model.encoder.load_state_dict(encoder_sd)

        latent_shape = (1, latent_channels, 64, 64)  # 512x512 output

        for i, seed in enumerate(SEEDS):
            g = torch.Generator().manual_seed(seed)
            latent = torch.randn(latent_shape, generator=g, dtype=torch.float32)
            with torch.no_grad():
                decoded = model.decoder(latent).clamp(0, 1)

            latent_nhwc = latent.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
            decoded_nhwc = decoded.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)

            save_file(
                {"latent": latent_nhwc},
                REFERENCE_DIR / f"{variant_name}_latent_{i:03d}.safetensors",
            )
            save_file(
                {"image": decoded_nhwc},
                REFERENCE_DIR / f"{variant_name}_decoded_{i:03d}.safetensors",
            )
            img_uint8 = (decoded[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(REFERENCE_DIR / f"{variant_name}_decoded_{i:03d}.png")

        print(f"  wrote 5 latents + 5 decoded images for {variant_name}")

        # === Encode the source image to a reference latent ===
        from PIL import Image as PILImage

        ref_img_path = Path(__file__).parent / "reference" / "_source_image.png"
        if ref_img_path.exists():
            src_pil = PILImage.open(ref_img_path).convert("RGB").resize((256, 256))
            # to_tensor: HWC uint8 -> NCHW float32 [0, 1]
            src_np = np.array(src_pil).astype(np.float32) / 255.0  # (256, 256, 3)
            src_tensor = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
            with torch.no_grad():
                encoded = model.encoder(src_tensor)
            encoded_nhwc = encoded.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
            save_file(
                {"latent": encoded_nhwc},
                REFERENCE_DIR / f"{variant_name}_encoded_001.safetensors",
            )
            print(f"  wrote encoded latent for {variant_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
