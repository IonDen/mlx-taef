"""Stable Diffusion (SD1.x / SDXL) model kernels."""

from mlx_taef.kernels._conversion import UpstreamTwoFile
from mlx_taef.kernels._types import ArchSpec, LatentSpec, ModelKernel, WeightSource

TAESD2D = ArchSpec(name="taesd2d")

TAESD = ModelKernel(
    name="taesd",
    arch=TAESD2D,
    conversion=UpstreamTwoFile(),
    latent=LatentSpec(channels=4),
    source=WeightSource(
        repo="madebyollin/taesd",
        decoder_filename="taesd_decoder.safetensors",
        encoder_filename="taesd_encoder.safetensors",
        revision="614f76814bbe30edbe2e627ace1c2234c81a2c0e",
        decoder_sha256="f0fb51dd10d41c26612c070fa0b52ea0215a5ff90792134b4971109dd713c019",
        encoder_sha256="160d90c61c3a5ce50fe2cbe6404b3429f5763772d61b38523cc37e1525b4e19f",
    ),
    integration=None,
    memory_cap_hint_gb=None,
)

TAESDXL = ModelKernel(
    name="taesdxl",
    arch=TAESD2D,
    conversion=UpstreamTwoFile(),
    latent=LatentSpec(channels=4),
    source=WeightSource(
        repo="madebyollin/taesdxl",
        decoder_filename="taesdxl_decoder.safetensors",
        encoder_filename="taesdxl_encoder.safetensors",
        revision="b20258aaef75ef61e659c1e0f14f251cf0ad153e",
        decoder_sha256="f6013131e7eb412ef20113f1acc2ea7d3e47e53196ca0530fa65d9b61d814b61",
        encoder_sha256="9f37c0b28f72ec4ca835dc7dbf05255bdf323cde8cf12a304674f106466c98ef",
    ),
    integration=None,
    memory_cap_hint_gb=None,
)
