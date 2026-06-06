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
    ),
    integration=None,
    memory_cap_hint_gb=None,
)
