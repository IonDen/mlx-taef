# Live preview

`mlx-taef` decodes the in-flight latent at every step, so you watch the image form instead of
waiting for the full VAE at the end.

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-taef/main/docs/assets/live-preview.gif" alt="TAEF1 live preview of a FLUX.1-dev generation" width="100%">
</p>

A FLUX.1-dev generation on an M1 Max, previewed step by step with the TAEF1 live-preview
callback. Every frame except the last is the tiny decoder; the final frame is the full VAE.
Wire previews into your own run with `LivePreviewCallback`; the [`examples/`](examples/)
directory has runnable scripts.
