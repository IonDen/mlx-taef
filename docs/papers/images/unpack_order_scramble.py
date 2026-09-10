"""Figure generator for docs/papers/the-latent-in-the-callback-is-not-the-latent-the-decoder-wants.md.

Packs a synthetic 32-channel latent the way mflux's FLUX.2 generator does, then unpacks it twice:
once with the FLUX.2 order (Flux2VAE._unpatchify_latents) and once with the Ideogram 4 order
(the reshape/transpose from Ideogram4LatentCreator.unpack_latents, shift/scale omitted so only the
layout differs). No weights, no model download, CPU only. Run from a checkout with the mflux extra:

    uv run --extra mflux python docs/papers/images/unpack_order_scramble.py

Verified against mflux 0.19.1.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from PIL import Image, ImageDraw, ImageFont

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).with_name("unpack-order-scramble.png")
)

H = W = 96  # spatial size of the 32-channel latent (image would be 8x = 768 px)


def picture(seed_shift: int = 0) -> np.ndarray:
    """An RGB test card: gradient, disc, bars, checker corner. Returns (3, H, W) in [-1, 1]."""
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for x in range(W):
        for y in range(H):
            img.putpixel((x, y), (int(255 * x / W), int(255 * y / H), 128))
    d.ellipse([20, 20, 76, 76], fill=(240, 60, 40), outline=(255, 255, 255), width=2)
    for i in range(6):
        d.rectangle(
            [4 + i * 6, 80, 8 + i * 6, 92], fill=(255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        )
    for x in range(72, 96, 4):
        for y in range(0, 24, 4):
            d.rectangle(
                [x, y, x + 3, y + 3],
                fill=(20, 20, 20) if ((x + y) // 4) % 2 == 0 else (230, 230, 230),
            )
    d.rectangle([40, 40, 56, 56], fill=(30, 200, 80))
    a = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    a = np.roll(a, seed_shift, axis=(0, 1))
    return a.transpose(2, 0, 1)


# 32 channels: channels 0-2 carry the test card; the rest carry shifted / inverted copies so that
# any channel mix-up is visible rather than blank.
chans = []
for c in range(32):
    p = picture(seed_shift=(c * 7) % H)
    if c % 3 == 1:
        p = -p
    if c % 3 == 2:
        p = p[:, ::-1, :]
    chans.append(p[c % 3])
latent = np.stack(chans)[None]  # (1, 32, H, W), the plain VAE latent
lat = mx.array(latent)

# The generator's own packing (FLUX.2 forward pack: channel-major, sub-pixel minor).
packed_nchw = Flux2LatentCreator.patchify_latents(lat)  # (1, 128, H/2, W/2)
packed_seq = Flux2LatentCreator.pack_latents(packed_nchw)  # (1, seq, 128)

# Route A: the matching unpack (Flux2VAE._unpatchify_latents on the NCHW form).
back_nchw = Flux2LatentCreator.unpack_latents(packed_seq, H * 8, W * 8)  # (1, 128, H/2, W/2)
route_a = Flux2VAE._unpatchify_latents(back_nchw)  # (1, 32, H, W)

# Route B: the sibling's unpack order (Ideogram4LatentCreator.unpack_latents layout only:
# reshape (B, gh, gw, 2, 2, C) -> transpose (0, 5, 1, 3, 2, 4); shift/scale omitted on purpose).
b, seq, ch = packed_seq.shape
gh, gw = H // 2, W // 2
x = packed_seq.reshape(b, gh, gw, 2, 2, ch // 4)
x = x.transpose(0, 5, 1, 3, 2, 4)
route_b = x.reshape(b, ch // 4, gh * 2, gw * 2)

ra = np.array(route_a)
rb = np.array(route_b)
print("route A exact:", np.array_equal(ra, latent))
print(
    "route B exact:", np.array_equal(rb, latent), "max abs err:", float(np.abs(rb - latent).max())
)
print("route B shape:", rb.shape)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Map channels 0-2 of a (1, 32, H, W) latent to a uint8 (H, W, 3) image."""
    a = arr[0, :3].transpose(1, 2, 0)
    return ((np.clip(a, -1, 1) + 1) * 127.5).astype(np.uint8)


scale = 3
panels = [
    ("Input latent (channels 0-2 as RGB)", to_rgb(latent)),
    ("Unpacked with the FLUX.2 order", to_rgb(ra)),
    ("Unpacked with the Ideogram 4 order", to_rgb(rb)),
]
pw, ph = W * scale, H * scale
pad, top = 16, 30
canvas = Image.new(
    "RGB", (len(panels) * pw + (len(panels) + 1) * pad, ph + top + pad), (24, 24, 24)
)
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
except Exception:
    font = ImageFont.load_default()
for i, (title, im) in enumerate(panels):
    x0 = pad + i * (pw + pad)
    canvas.paste(Image.fromarray(im).resize((pw, ph), Image.NEAREST), (x0, top))
    d.text((x0, 9), title, fill=(230, 230, 230), font=font)
canvas.save(OUT)
print("saved")
