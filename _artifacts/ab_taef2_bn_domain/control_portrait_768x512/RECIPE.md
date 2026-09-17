# Control latent recipe

The 512×512 result one directory up uses the committed, sha-pinned fixture
`tests/fixtures/showcase_latents/flux2_klein_base_4b.safetensors`, a 4-step FLUX.2 Klein base 4B
latent that is deliberately under-denoised (it is a timing fixture). This control repeats the
measurement on a fully denoised, non-square image with a different subject.

The control latent itself is not committed. Capture it, then run the A/B on it:

```
uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b \
    --out-dir /tmp/ab_control --height 512 --width 768 --num-steps 28 --guidance 4.0 --seed 7 \
    --prompt "portrait photo of an elderly fisherman in a yellow raincoat, harbour at dawn, detailed skin, 50mm"
mv /tmp/ab_control/flux2_klein_base_4b.safetensors /tmp/ab_control/flux2_klein_base_4b_portrait_768x512.safetensors
uv run python scripts/ab_taef2_bn_domain.py \
    --latent /tmp/ab_control/flux2_klein_base_4b_portrait_768x512.safetensors \
    --out-dir _artifacts/ab_taef2_bn_domain/control_portrait_768x512
```

The committed run used M1 Max 32 GB, macOS 27.0, mflux 0.19.1, MLX 0.32.2, int4 weights. Its latent
had sha256 `bfee11ad6fa56692e38fe0f2b9a97084b04b6c822b1016232617e818656d5b29`, recorded in every
result file here. MLX generation is not bit-reproducible across processes and versions, so a fresh
capture gives a slightly different latent and scores that differ in the third decimal; the gap
between the two TAEF2 readings is two orders of magnitude larger than that.
