# Control latent recipe

The 512×512 result one directory up uses the committed, sha-pinned fixture
`tests/fixtures/showcase_latents/flux2_klein_base_4b.safetensors`, a 4-step FLUX.2 Klein base 4B
latent that is deliberately under-denoised (it is a timing fixture). This control repeats the
measurement on a fully denoised, non-square image with a different subject.

The control latent itself is not committed. Capture it, then run the A/B on it:

```
WORK=$(mktemp -d)
uv run python scripts/_capture_latent.py --variant flux2-klein-base-4b \
    --out-dir "$WORK" --height 512 --width 768 --num-steps 28 --guidance 4.0 --seed 7 \
    --prompt "portrait photo of an elderly fisherman in a yellow raincoat, harbour at dawn, detailed skin, 50mm"
mv "$WORK/flux2_klein_base_4b.safetensors" "$WORK/flux2_klein_base_4b_portrait_768x512.safetensors"
uv run python scripts/ab_taef2_bn_domain.py \
    --latent "$WORK/flux2_klein_base_4b_portrait_768x512.safetensors" \
    --out-dir "$WORK/ab"
```

Point `--out-dir` at this directory instead if you mean to replace the committed result; the script
stages its work and only overwrites these files once all three decodes and the scoring succeed.

The committed run used M1 Max 32 GB, macOS 27.0, mflux 0.19.1, MLX 0.32.2, int4 weights. Its latent
had sha256 `bfee11ad6fa56692e38fe0f2b9a97084b04b6c822b1016232617e818656d5b29`, recorded in every
result file here. A capture on another MLX or mflux version, or another chip, may not reproduce
that latent bit for bit. Compare the sha256 to see whether you have the same one, and expect small
score differences if you do not; the gap between the two TAEF2 readings is far larger than that.
