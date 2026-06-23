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


def _generate_qwen_references() -> None:
    """Generate qwen-image (taew2.1) reference fixtures + converted MLX weights.

    Uses the vendored upstream taehv oracle (fp32) on the sha256-verified canonical checkpoint.
    Kept separate from the TAESD loop because taehv has a distinct temporal forward + encode path.
    Decode/encode run at T=1 (a single still image) with H,W>1 latents.
    """
    import mlx.core as mx
    from safetensors.numpy import load_file as load_np
    from safetensors.torch import load_file as load_torch

    sys.path.insert(0, str(Path(__file__).parent / "_third_party"))
    sys.path.insert(0, str(Path(__file__).parent))
    from _taehv_canonical import canonical_taew21_path
    from taehv import TAEHV

    from mlx_taef.convert import _build_mlx_state_dict, _flatten_module_param_shapes
    from mlx_taef.kernels._arch import build_arch
    from mlx_taef.kernels._conversion import TaehvCombined

    converted_dir = Path(__file__).parent / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    ckpt = canonical_taew21_path()

    # Oracle in fp32 (canonical is fp16; the MLX port also runs fp32 — see the review dtype note).
    model = TAEHV(checkpoint_path=None).eval().float()
    sd = {k: v.float() for k, v in load_torch(str(ckpt)).items()}
    model.load_state_dict(model.patch_tgrow_layers(sd))

    lh = lw = 16  # 16x16 latent -> 128x128 RGB
    for i, seed in enumerate(SEEDS):
        g = torch.Generator().manual_seed(seed)
        latent = torch.randn(1, 1, 16, lh, lw, generator=g, dtype=torch.float32)  # N,T,C,H,W
        with torch.no_grad():
            decoded = model.decode_video(latent, parallel=True, show_progress_bar=False)
        latent_nhwc = latent[:, 0].permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
        decoded_nhwc = decoded[:, 0].permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
        save_file({"latent": latent_nhwc}, REFERENCE_DIR / f"qwen-image_latent_{i:03d}.safetensors")
        save_file(
            {"image": decoded_nhwc}, REFERENCE_DIR / f"qwen-image_decoded_{i:03d}.safetensors"
        )
        img_uint8 = (decoded_nhwc[0] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(REFERENCE_DIR / f"qwen-image_decoded_{i:03d}.png")

    # Encode reference from the shared source image.
    src = Image.open(REFERENCE_DIR / "_source_image.png").convert("RGB").resize((256, 256))
    src_np = np.array(src).astype(np.float32) / 255.0
    src_t = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # N,T,C,H,W
    with torch.no_grad():
        encoded = model.encode_video(src_t, parallel=True, show_progress_bar=False)
    encoded_nhwc = encoded[:, 0].permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    save_file({"latent": encoded_nhwc}, REFERENCE_DIR / "qwen-image_encoded_001.safetensors")

    # Converted MLX weights (fp32) via the kernel conversion path.
    full = load_np(str(ckpt))
    for role in ("decoder", "encoder"):
        raw = TaehvCombined._select_role(full, role)
        arch = build_arch("taehv", role=role, latent_channels=16, midblock_gn=False)
        converted = _build_mlx_state_dict(raw, expected_shapes=_flatten_module_param_shapes(arch))
        mx.save_safetensors(str(converted_dir / f"qwen-image_{role}.safetensors"), converted)
    print("  wrote qwen-image: 5 latents + 5 decoded + 1 encoded + 2 converted weight files")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate parity reference fixtures.")
    parser.add_argument(
        "--only",
        help="Generate only this variant's fixtures (e.g. 'qwen-image'); leaves others untouched.",
    )
    args = parser.parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if args.only == "qwen-image":
        _generate_qwen_references()
        return 0

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
