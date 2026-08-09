# Live preview

`mlx-taef` decodes the in-flight latent at every step, so you watch the image form instead of
waiting for the full VAE at the end.

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-taef/main/docs/assets/live-preview.gif" alt="Side-by-side animated GIF: TAEF1 per-step live previews animating on the left against the finished full-VAE decode held static on the right, with a step-count caption." width="100%">
</p>

A FLUX.1-dev generation on an M1 Max, previewed step by step with the TAEF1 live-preview
callback: the tiny decoder's previews animate on the left while the finished full-VAE decode
holds static on the right, with a step-count caption. Wire previews into your own run with
`LivePreviewCallback`; the [`examples/`](examples/) directory has runnable scripts.
