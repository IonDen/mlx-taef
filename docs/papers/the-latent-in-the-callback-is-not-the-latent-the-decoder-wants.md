# The latent in the callback is not the latent the decoder wants

*A research memo from `mlx-taef`: nine mflux image models, three VAE families, six in-loop
tensor layouts, and why sharing a decoder does not mean sharing a latent contract*

> 📄 [Read on the website](https://ineshin.space/papers/the-latent-in-the-callback-is-not-the-latent-the-decoder-wants/) — same paper, formatted for reading.

A live preview during a diffusion run is a small decoder applied to the tensor the sampler is
still working on. [`mlx-taef`](https://github.com/IonDen/mlx-taef) ships that decoder for five
[mflux](https://github.com/mflux-community/mflux) model families on Apple Silicon. Adding each
family taught the same lesson from a different side: the tensor a generator hands its in-loop
callback is not the tensor its VAE decodes. Between the two sit a packing, a sub-pixel transpose,
a spatial convention, and a normalization, and all four live in the *generator*, not the decoder.
Two models can decode through byte-identical VAE weights and still hand a callback different
tensors, and two models that hand it the same shape can fold the sub-pixels into that shape in
different orders. A wrong rank or channel count raises. A wrong fold order or a wrong affine
map does not, and those are the mistakes this memo is about.

This memo is documentary. Every per-model claim is read from the mflux source at the public
[`v.0.19.1`](https://github.com/mflux-community/mflux/tree/v.0.19.1) tag and from `mlx-taef` at
[`v0.8.1`](https://github.com/IonDen/mlx-taef/tree/v0.8.1); every mflux and `mlx-taef` source
link below is pinned to one of those two tags, and links into other projects' code are pinned
to a commit. No image was generated and nothing was benchmarked for this write-up. The
quality numbers quoted from the `mlx-taef` changelog are source-reported, measured earlier on a
10-core Apple M1 Max with 32 GB of unified memory. Two file-level checks were run for this memo
and are dated as such: the sha256 digests in section 1 were read from the Hugging Face Hub API on
2026-09-09, and the weight comparison in section 5 was re-run on the same day on the CPU with
NumPy. The figure in section 3 is a synthetic tensor pushed through two reshape orders; it needs
no weights, and its generator script is committed beside it.

## 1. The seductive fact: same VAE, same weights

Several of mflux's image models decode through one VAE class. The three families below cover all
nine models in this memo:

| VAE class in mflux | Latent channels | Spatial factor | Models that construct it |
|---|---|---|---|
| [`Flux2VAE`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux2/model/flux2_vae/vae.py) | 32 | 8 | FLUX.2 Klein, [Lens](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/lens/variants/txt2img/lens_image.py), [ERNIE-Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/ernie_image/ernie_image_initializer.py), [Ideogram 4](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/ideogram4/ideogram4_initializer.py) |
| FLUX.1 [`VAE`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux/model/flux_vae/vae.py) and the Z-Image [`VAE`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/z_image/model/z_image_vae/vae.py) (same constants) | 16 | 8 | FLUX.1 dev / schnell, [Boogu Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/boogu/variants/txt2img/boogu_image.py), Z-Image / Z-Image-Turbo |
| [`QwenVAE`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/qwen/model/qwen_vae/qwen_vae.py) (the Wan 2.1 VAE, per [madebyollin's README](https://github.com/madebyollin/taesd/blob/e87efbcfc5298d84986b5d9280f40d358b6a228d/README.md)) | 16 | 8 | Qwen-Image / Qwen-Image-Edit, [Krea 2](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/krea2/krea2_initializer.py) |

The class is only half of the story. The other half is whether the trained weights are the same,
and for the `Flux2VAE` group the Hub answers that with a hash. The table lists the `vae/`
safetensors file each model's Hub repository ships, as reported by the Hub API on 2026-09-09:

| Repository | File | Bytes | sha256 (prefix) |
|---|---|---|---|
| `black-forest-labs/FLUX.2-klein-4B` | `vae/diffusion_pytorch_model.safetensors` | 168,120,878 | `ca70d220` |
| `baidu/ERNIE-Image` | `vae/diffusion_pytorch_model.safetensors` | 168,120,878 | `ca70d220` |
| `ideogram-ai/ideogram-4-fp8` | `vae/diffusion_pytorch_model.safetensors` | 168,120,878 | `925ce3a0` |
| `black-forest-labs/FLUX.2-dev` | `vae/diffusion_pytorch_model.safetensors` | 336,213,556 | `d64f3a68` |
| `Comfy-Org/Lens` | `vae/flux2-vae.safetensors` | 336,213,556 | `d64f3a68` |
| `Qwen/Qwen-Image` | `vae/diffusion_pytorch_model.safetensors` | 253,806,966 | `0c8bc8b7` |
| `Qwen/Qwen-Image-2512` | `vae/diffusion_pytorch_model.safetensors` | 253,806,966 | `0c8bc8b7` |
| `krea/Krea-2-Turbo` | `vae/diffusion_pytorch_model.safetensors` | 507,591,892 | `ab1b6110` |
| `Tongyi-MAI/Z-Image-Turbo` | `vae/diffusion_pytorch_model.safetensors` | 167,666,902 | `f5b59a26` |
| `Boogu/Boogu-Image-0.1-Turbo` | `vae/diffusion_pytorch_model.safetensors` | 335,306,212 | `8c717328` |

ERNIE-Image's VAE file is byte-identical to FLUX.2 Klein 4B's, which is what
the [ERNIE-Image technical report](https://arxiv.org/abs/2605.25347) says in its introduction:
"We adopt the FLUX.2 VAE … as our variational autoencoder". Lens ships the FLUX.2-dev file
byte-for-byte, and mflux's Lens loader does not even use it: it
[loads the VAE from the Klein 4B repository](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/lens/variants/txt2img/lens_image.py#L32)
by name. Ideogram 4's file has the same size as Klein 4B's to the byte and a different digest.
Ideogram's own reference code
[calls its autoencoder module "Flux2 KL autoencoder"](https://github.com/ideogram-oss/ideogram4/blob/d17489da7c839bbc2052b6c7e8c46ece65cb78ad/src/ideogram4/autoencoder.py),
so the architecture is not in doubt, but whether the trained weights are Klein's stays an
inference: the repository is gated, and the digest says the file is not the same bytes. The other
families are looser still: Krea 2 reuses the `QwenVAE` class and its weight mapping but ships its
own file (twice the size of Qwen-Image's, consistent with fp32 against bf16), and Z-Image reuses
FLUX.1's normalization constants (`scaling_factor` 0.3611, `shift_factor` 0.1159) with weights
from its own repository.

So a preview library has real reuse to harvest. madebyollin's README already
[lists the pairings](https://github.com/madebyollin/taesd/blob/e87efbcfc5298d84986b5d9280f40d358b6a228d/README.md):
Z-Image with the FLUX.1 tiny decoder (TAEF1), Qwen-Image with the Wan 2.1 tiny decoder (taew2.1).
`mlx-taef` previews Z-Image and Krea 2 that way with no new weights in either case, and its
changelog records the measured agreement with each model's full VAE at SSIM
[0.94](https://github.com/IonDen/mlx-taef/blob/v0.8.1/CHANGELOG.md) for Z-Image and
[0.9678](https://github.com/IonDen/mlx-taef/blob/v0.8.1/CHANGELOG.md) for Krea 2. The trap is the
inference that runs the other way: "same decoder, so the same unpack will do."

## 2. What the callback actually hands you

mflux calls every registered in-loop callback once per denoising step with the sampler's current
tensor. The table lists that tensor for each model, read from the line where the generator calls
`ctx.in_loop`, together with the two conventions a decoder needs to know and does not receive:
where the sub-pixel unpack happens and how the values are normalized. `h` and `w` are the image
height and width in pixels.

| Model | In-loop tensor | Spatial divisor | Sub-pixel unpack lives in | Normalization applied by |
|---|---|---|---|---|
| [FLUX.1](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux/variants/txt2img/flux.py#L124) | `(1, h/16 · w/16, 64)` packed | 16 | [`FluxLatentCreator.unpack_latents`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux/latent_creator/flux_latent_creator.py#L19) | the VAE, scalar scale and shift inside `decode` |
| [Qwen-Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/qwen/variants/txt2img/qwen_image.py#L130) | `(1, h/16 · w/16, 64)` packed | 16 | [`QwenLatentCreator`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/qwen/latent_creator/qwen_latent_creator.py), which delegates to FLUX.1's | the VAE, per-channel mean and std vectors inside `decode` |
| [FLUX.2 Klein](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux2/variants/txt2img/flux2_klein.py#L111) | `(1, h/16 · w/16, 128)` packed | 16 | the VAE, [`Flux2VAE._unpatchify_latents`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux2/model/flux2_vae/vae.py#L60) | the VAE, batch-norm running statistics (`vae.bn`, 128 channels) |
| [Lens](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/lens/variants/txt2img/lens_image.py#L118) | `(1, h/16 · w/16, 128)` packed | 16 | the VAE, as FLUX.2 | the VAE, as FLUX.2 |
| [ERNIE-Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/ernie_image/variants/txt2img/ernie_image.py#L101) | `(1, 128, h/16, w/16)`, no sequence axis | 16 | the VAE, as FLUX.2 | the VAE, as FLUX.2 |
| [Ideogram 4](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/ideogram4/variants/txt2img/ideogram4.py#L146) | `(1, h/16 · w/16, 128)` packed | 16 | [`Ideogram4LatentCreator.unpack_latents`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/ideogram4/latent_creator/ideogram4_latent_creator.py#L300) | the generator: 128 baked shift and scale constants; the VAE's `bn` buffers are never read |
| [Krea 2](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/krea2/variants/txt2img/krea2.py#L100) | `(1, 16, h/8, w/8)` | 8 | nowhere: [`unpack_latents` is the identity](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/krea2/latent_creator/krea2_latent_creator.py) (`pack_latents` only drops a frame axis) | the VAE, as Qwen-Image |
| [Z-Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/z_image/variants/z_image.py#L126) | `(16, 1, h/8, w/8)`, channels first, no batch axis, one temporal frame | 8 | nowhere: [unpack](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/z_image/latent_creator/z_image_latent_creator.py#L28) only moves the singleton axis | the VAE, scalar scale and shift (FLUX.1's values) |
| [Boogu Image](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/boogu/variants/txt2img/boogu_image.py#L96) | `(1, 16, h/8, w/8)` | 8 | nowhere | the VAE, as FLUX.1 |

Read down any column and the "shared VAE" grouping from section 1 dissolves. The four `Flux2VAE`
models arrive in three layouts (sequence-packed 128, channel-first 128 without a sequence axis,
and sequence-packed 128 with a different channel order) and two normalization schemes (the VAE's
own batch-norm buffers, or constants the generator carries and the VAE never sees). The three
16-channel FLUX.1-style models arrive as a packed sequence, as a plain NCHW tensor, and as a
channels-first tensor with no batch axis and a singleton frame axis. The Wan 2.1 pair arrives
packed for Qwen-Image and unpacked for Krea 2, so the same tiny decoder needs a sub-pixel unpack
for one and a bare transpose for the other.

The spatial divisor is a smaller trap with the same shape. A callback that derives the latent
grid from the generation config must divide by 16 for the packed models and by 8 for the rest;
`mlx-taef` [records the divisor per binding](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/kernels/_types.py)
and [derives the grid from mflux's config only when the binding is packed](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/integrations/mflux.py).
A wrong divisor raises on its own, because the sequence length stops matching the grid, and so
does a missing unpack (section 4). The permutation and normalization errors do not.

The table describes the producer. The consumer has expectations of its own, and they are not
the full VAE's. Each tiny decoder was distilled against a particular input domain. TAEF1 and
taew2.1 take the sampler-space latent as it is: madebyollin's README puts TAESD's latent scale
factor at 1, and `mlx-taef`'s
[Qwen-Image unpack](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/kernels/qwen.py)
deliberately applies no mean and std because the weights already account for them. TAEF2 is
the open case: `mlx-taef` applies the VAE's batch-norm inverse before it, while madebyollin's
own [reference wrapper](https://huggingface.co/madebyollin/taef2) plugs the decoder in behind a
default batch norm (mean 0, variance 1, so the inverse is the identity) and ComfyUI's TAEF2
decoder only unpatchifies. The repository has not measured which input domain the decoder was
distilled on, and the two readings differ by a per-channel factor of about 1.8, so this is a
value-space question the shape can never settle. A binding reconciles two things: the layout
and value domain the generator emits, and the layout and value domain the decoder in hand was
trained on. Applying the full VAE's per-channel mean and std before taew2.1 would be wrong, and
whichever side of the TAEF2 question is wrong is wrong in the same silent way.

None of this is documented as a contract. Each fact is read from the generator's own loop and
latent-creator source at a pinned version, and from the tiny decoder's training convention.
That is the method this memo argues for, and section 6 comes back to it.

## 3. The sub-pixel order: two models, one decoder, two packings

FLUX.2 Klein and Ideogram 4 are the sharpest case. Both decode through `Flux2VAE`, both hand the
callback a sequence-packed `(1, seq, 128)` tensor at one sixteenth of the image size, and both
fold a 2×2 sub-pixel block into the 128-channel axis. They fold it in different orders.

FLUX.2 packs channel-major. Its
[`patchify_latents`](https://github.com/mflux-community/mflux/blob/v.0.19.1/src/mflux/models/flux2/latent_creator/flux2_latent_creator.py#L8)
reshapes the 32-channel latent to `(B, 32, H/2, 2, W/2, 2)`, transposes to
`(B, 32, 2, 2, H/2, W/2)` and folds the 32, 2, 2 axes into one, so packed channel `k` holds
latent channel `c` at sub-pixel row `py` and column `px` with

```
k = 4·c + 2·py + px          (FLUX.2, ERNIE-Image, FLUX.1 and Qwen-Image alike)
```

This is the convention of the
[FLUX.2 reference autoencoder](https://github.com/black-forest-labs/flux2/blob/e80b84ed9f2a2346a97d2ee033eecf65cf685e31/src/flux2/autoencoder.py),
whose encoder rearranges `c (i pi) (j pj)` into `(c pi pj) i j`, channel outermost.
`Flux2VAE._unpatchify_latents` inverts exactly that, and FLUX.1's `pack_latents` uses the same
convention for its 64 channels. Ideogram 4 packs sub-pixel-major. Its `unpack_latents` reshapes
the packed axis as `(B, gh, gw, 2, 2, 32)` and transposes to `(B, 32, gh, 2, gw, 2)`, which means

```
k = 64·py + 32·px + c        (Ideogram 4)
```

and matches
[Ideogram's own reference pipeline](https://github.com/ideogram-oss/ideogram4/blob/d17489da7c839bbc2052b6c7e8c46ece65cb78ad/src/ideogram4/pipeline_ideogram4.py),
which mflux mirrors.

Apply the Ideogram order to a FLUX.2-packed tensor and every output value comes from a real
input value, just the wrong one. Output channel 0 becomes a 2×2 interleave of input channels 0,
8, 16 and 24, each sampled at its top-left sub-pixel; output channel `c′` in general draws from
input channels `⌊c′/4⌋ + {0, 8, 16, 24}`, with the sampled sub-pixel cycling through the four
positions as `c′` counts up. The figure shows it on a synthetic 32-channel latent whose first
three channels carry a test card. The left panel is the input, the middle panel is the input
packed by mflux's FLUX.2
functions and unpacked by `Flux2VAE._unpatchify_latents` (bit-exact with the input), and the
right panel is the same packed tensor unpacked with the Ideogram reshape and transpose:

![Three panels. Left: a synthetic test card (gradient, disc, bars, checker corner) used as channels 0 to 2 of a 32-channel latent. Middle: the same card after packing with mflux's FLUX.2 order and unpacking with the FLUX.2 order, identical to the left. Right: the same packed tensor unpacked with the Ideogram 4 order, a fine 2x2 pixel interleave that mixes four channels and keeps only the rough outline of the disc.](https://raw.githubusercontent.com/IonDen/mlx-taef/main/docs/papers/images/unpack-order-scramble.png)

*Synthetic evidence, not a model output: a 96×96×32 test latent through two unpack orders. The
right panel keeps the disc's outline and the checker corner's position, which is what would make
the failure easy to mistake for a noisy early step in a small preview. Generator script:
[`unpack_order_scramble.py`](https://github.com/IonDen/mlx-taef/blob/main/docs/papers/images/unpack_order_scramble.py).*

The two orders disagree even before the normalization does, and the normalization disagrees
too: Ideogram's 128 shift and scale constants are baked into its latent creator and applied
before the unpack, while FLUX.2 de-normalizes with the VAE's batch-norm buffers. These are not
the same numbers kept in two places. Against the batch-norm statistics stored in the Klein 4B
VAE file, Ideogram's shift constants differ by up to 0.43 and its scale constants by up to 16%
(compared on 2026-09-09; Ideogram's own VAE file could not be read for the same comparison,
since its repository is gated). Diffusers'
[Ideogram 4 pipeline](https://github.com/huggingface/diffusers/blob/9602fc526451382b48aa672a40dae8640e14df69/src/diffusers/pipelines/ideogram4/pipeline_ideogram4.py)
reads shift and scale from its VAE file's batch-norm buffers rather than from constants, so
whether the two implementations agree depends on what that gated file carries. A preview built
by borrowing FLUX.2's unpack for Ideogram would apply the wrong affine map to the wrong channel
order and decode the result through weights that are, at best, FLUX.2's. Nothing in that path
can raise.

`mlx-taef` does not ship an Ideogram 4 preview. The scramble above is a prediction from the two
source files, made concrete with a synthetic tensor, not a captured Ideogram frame; the point of
the figure is that the prediction needs no weights to check.

## 4. Which failures here are silent

Each convention in section 2 has a failure mode. The ones that change a tensor's rank or
channel count raise. The ones that keep the shape and permute or rescale the values do not,
and those are the ones a shape check cannot see.

The normalization fails quietly. FLUX.2's in-loop tensor is batch-norm-normalized with
statistics that live in the VAE. `mlx-taef` reads them off the model instance when the caller
passes one, and
[falls back to identity](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/integrations/mflux.py)
otherwise, exposing which path won as `resolved_bn`. The repository documents the identity path
as a degradation, structurally right with shifted colours, though that description was never
measured against the auto path, and section 2 notes that other TAEF2 integrations take the
identity path on purpose. Before the auto-extraction existed that was the default behaviour, and the
[manual-verification recipe](https://github.com/IonDen/mlx-taef/blob/v0.8.1/docs/manual-verification.md)
still lists it as the third of three precedence levels because a caller without the model
handy will hit it. The same family of error reaches upstream code: a Diffusers review of the
ERNIE-Image pipeline
([issue 13577](https://github.com/huggingface/diffusers/issues/13577), closed 2026-05-06) found
it de-normalizing with a hard-coded epsilon of 1e-5 while the VAE's configured value was 1e-4,
a value-space drift no shape check would report, and `mlx-taef` had to forward the VAE's actual
epsilon into its own unpack in v0.5.0 for the same reason.

A missing unpack fails loudly, which is the useful contrast. `mlx-taef`'s first FLUX.1 preview
[fed the packed `(1, seq, 64)` tensor straight to the decoder](https://github.com/IonDen/mlx-taef/blob/v0.8.1/CHANGELOG.md)
because a guard that looked for a 16-channel axis never matched on the packed shape and let the
tensor fall through. That tensor has the wrong rank, and the decoder's first convolution rejects
it, so the preview did not come out wrong; it did not come out at all. The changelog's "wrong
previews" is loose wording for a broken feature. It shipped that way from the first release on
2026-05-13 until v0.3.0 on 2026-06-06, and the fix came with the test described in section 6.
A structural error is caught by the first layer that touches the tensor. A value error is
caught by nothing.

The transpose fails quietly: section 3's scramble is a plausible image with no exception.

The layout family is the one case with a guard. Hand Z-Image's `(16, 1, h, w)` tensor to a
callback that expects `(1, 16, h, w)` and the channel axis becomes the batch axis. `mlx-taef`
rejects this by
[checking the exact rank and leading dimensions](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/kernels/zimage.py),
which is a check written after the fact, not a property of the tensor.

The repository's own verification document states the limit plainly: automated tests
[use a fake-shaped latent and cannot catch value-space drift](https://github.com/IonDen/mlx-taef/blob/v0.8.1/docs/manual-verification.md).
A shape test proves the plumbing is connected. It does not prove the water is going the right
way. The same distinction, between a test that can pass on structure alone and a test that
constrains values, is the subject of a companion memo from the sibling library,
[Why byte-exact parity is a poor MLX integration oracle](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-byte-exact-parity-is-a-poor-mlx-integration-oracle.md).

## 5. The second silent trap: weight mirrors are not the weights

The latent contract has a twin on the weights side. `mlx-taef`'s Qwen-Image and Krea 2
previews use madebyollin's taew2.1 tiny decoder, whose canonical weights are published
[on GitHub only](https://github.com/madebyollin/taehv/blob/0ad83bb8fdc48e9e94138704e939d500a3b43660/safetensors/taew2_1.safetensors).
A Hugging Face repository, `lightx2v/Autoencoders`, carries a file with the same name. On
2026-06-23, while the Qwen-Image port was being built, the two were compared and the mirror was
rejected. The comparison was repeated for this memo on 2026-09-09:

| | Canonical (`madebyollin/taehv`, `safetensors/taew2_1.safetensors`) | Mirror (`lightx2v/Autoencoders`, `taew2_1.safetensors`) |
|---|---|---|
| Bytes | 22,642,902 | 22,642,902 |
| sha256 (prefix) | `04766eac` | `79482981` |
| Tensors | 128 | 128, same names, same shapes |
| dtype | float16 | float16 |
| Elements that differ | — | 9,843,197 of 11,315,539 |
| Max absolute difference | — | 3.56 |

Same file size to the byte, same key set, same shapes, same dtype, and 87% of the values differ.
Whether the mirror is a different fine-tune, a conversion error, or an unrelated checkpoint
under a familiar name is not something this memo can tell, and nothing here says how well it
decodes. What the comparison does establish is narrower and sufficient: a loader keyed on
filename, tensor names and shapes would accept it without a word, and every parity fixture and
SSIM gate that was run against the canonical checkpoint says nothing about this one. The file
format does not help: safetensors carries no checksum of its own, and a request to add one
([issue 220](https://github.com/safetensors/safetensors/issues/220)) was closed in 2023 without
a change. `mlx-taef` therefore pins every weight source by
[an immutable revision and a sha256](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/kernels/qwen.py)
and refuses to load on a mismatch, and its Qwen-Image source is a re-host of the canonical file
verified against that digest. Names have stopped being enough elsewhere too: a ComfyUI issue
about a different, optimized preview decoder from the same repository
([issue 13366](https://github.com/Comfy-Org/ComfyUI/issues/13366)) identifies the file it means
by its sha256. The reproduction of the table needs one `curl` per file, `shasum -a 256`, and a
six-line NumPy loop over `safetensors.numpy.load_file`; no GPU is involved.

## 6. The discipline that works

Three practices came out of the five integrations, in the order they were learned.

The first is to read the generator's source at the pinned version and never infer a contract
from a sibling. Every unpack in `mlx-taef` names the mflux function it mirrors, and the Krea 2
kernel also records the file and the version it was read from, which is the standard the others
should be held to. Where upstream exposes the inverse function, the test suite calls it as an
oracle:
[`unpack_flux1_latent`](https://github.com/IonDen/mlx-taef/blob/v0.8.1/src/mlx_taef/kernels/flux.py)
is asserted exactly equal to mflux's own `FluxLatentCreator.unpack_latents` on an `arange` tensor
([test](https://github.com/IonDen/mlx-taef/blob/v0.8.1/tests/test_integrations_kernels.py)),
and the Qwen-Image unpack is checked against `QwenLatentCreator.unpack_latents` on a seeded
random tensor within a 1e-5 tolerance
([test](https://github.com/IonDen/mlx-taef/blob/v0.8.1/tests/test_unpack_qwen.py)). Of the two,
the `arange` form is the stronger and the cheaper: every element is its own index, so any
permutation shows up as an exact mismatch, and the Qwen test would lose nothing by adopting it.

For FLUX.2 the upstream inverse is split across
`Flux2LatentCreator.unpack_latents` and `Flux2VAE._unpatchify_latents`; the v0.8.1 test predates
composing the two as an oracle and
[pins the routing of four specific elements](https://github.com/IonDen/mlx-taef/blob/v0.8.1/tests/test_integrations_kernels.py)
instead, and the figure script in section 3 is exactly the composed oracle that test should
adopt. Krea 2's kernel docstring records the file paths and function names that establish the
identity pack, and it was written after reading them, not after noticing that Krea 2 shares
Qwen-Image's VAE.

The second is to pin weights by digest, not by name. Section 5.

The third is to climb the reuse ladder one rung at a time. The five shipped bindings and the
four candidates sort into three rungs. Zero new code: Lens should preview through the FLUX.2
binding unchanged, since its loop, its layout and its VAE weights are Klein 4B's; that is a
source-read verdict awaiting a validation run, not a shipped feature. A bespoke unpack on
existing weights: Krea 2 today, and, if they were built, ERNIE-Image on TAEF2, Boogu on TAEF1
(its `(1, 16, h/8, w/8)` tensor is Krea 2's layout, not Z-Image's, and the Z-Image guard rejects
it; Boogu also ships its own VAE file, so TAEF1's fidelity there would be a measurement, not an
assumption), and Ideogram 4 only if its differently-hashed VAE file turns out to be FLUX.2's.
New weights: FLUX.2 Klein and Qwen-Image. The rung is decided by the in-loop tensor and the
normalization, never by which VAE class the model constructs.

Upstream's own history shows the contract has a runtime dimension too. `QwenVAE` keeps its
per-channel latent mean and std as class attributes. Until
[mflux PR #478](https://github.com/mflux-community/mflux/pull/478) (merged 2026-08-10) they were
built as lazy MLX arrays at import time, which, as the PR explains, MLX 0.31 and later pin to the
stream of the importing thread; any host that imports on one thread and generates on another,
ComfyUI being the case reported, crashed at the first evaluation. The PR notes that Krea 2
inherited the crash the moment it started constructing `QwenVAE`. Reuse propagates a contract
together with its defects, which is one more reason to know exactly what is being reused.

## 7. What transfers

Nothing above is specific to MLX or to mflux. Every framework that exposes a per-step hook hands
out a tensor from inside the sampler, and the first question is which one. mflux's `in_loop`
and Diffusers' `callback_on_step_end` pass the current, still-noisy working state; ComfyUI's
[preview callback](https://github.com/Comfy-Org/ComfyUI/blob/3216c62e9962c3babd28a4dfea6e5aef50b8fe16/latent_preview.py)
receives both the state `x` and the denoised estimate `x0` and decodes `x0`. Two tensors with
the same shape, packing and normalization are still not interchangeable if one is the noisy
state and the other is the model's current guess at the clean image, and a preview of the wrong
one is another picture that looks plausible. In every framework the tensor's meaning, layout,
spatial convention and normalization are properties of the pipeline that produced it, and the
VAE it will eventually reach says nothing about them. Krea 2 makes the point on its own: Diffusers'
[Krea 2 pipeline](https://github.com/huggingface/diffusers/blob/3993de59e37344d92aa24ec25bdc39413157b744/src/diffusers/pipelines/krea2/pipeline_krea2.py)
runs its loop on a sequence-packed `(B, seq, 64)` tensor, mflux runs the same model on an
unpacked `(1, 16, h/8, w/8)` one, and the VAE is the same in both. The
[Diffusers callback guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/callback)
shows a preview for a 4-channel SDXL latent and says nothing about packing; the first public
FLUX preview from that hook
([discussion 6991](https://github.com/huggingface/diffusers/discussions/6991)) had to call the
pipeline's private unpack and re-apply the VAE's scale and shift by hand.

The nearest existing answer to "what is a latent contract" is ComfyUI's
[`latent_formats.py`](https://github.com/Comfy-Org/ComfyUI/blob/00d34d92fe0afbfbab3893ebbab2d5d70f5e9882/comfy/latent_formats.py),
a per-model object carrying channel count, spatial divisor, scale and shift, and the name of
the tiny decoder to preview with.
[`supported_models.py`](https://github.com/Comfy-Org/ComfyUI/blob/3216c62e9962c3babd28a4dfea6e5aef50b8fe16/comfy/supported_models.py)
assigns ERNIE-Image, Ideogram 4 and Lens the same `Flux2` format, and that is not an oversight.
ComfyUI settles the packing question by fiat: the sampler and the preview only ever see the
channel-packed grid, and each model's adapter converts to and from its own token order inside
the transformer, as the
[Ideogram 4 adapter](https://github.com/Comfy-Org/ComfyUI/blob/3216c62e9962c3babd28a4dfea6e5aef50b8fe16/comfy/ldm/ideogram4/model.py)
does with the same `(B, 32, 2, 2, gh, gw)` reshape this memo derives. The model-specific part of
the contract lives with the model, and the shared format is a normalized boundary. That is the
same discipline section 6 argues for, applied one layer down, and it is why a ComfyUI preview
needs no per-model unpack while an mflux callback does: mflux hands out the transformer's own
token layout. The cost is that the packing knowledge is procedural, inside each adapter, rather
than declared where a reader can compare it across models, which is what the table in section 2
tries to supply.

The transferable rule is short. A latent contract is what the tensor represents, plus layout,
normalization and spatial convention, read from the generator's source at a pinned version, and
matched against the input domain the decoder in hand was trained on. Which VAE class the
generator constructs settles none of that, and weight identity is a hash or it is an assumption.

---
*Denis Ineshin · 2026-09-10 · [ineshin.space](https://ineshin.space)*
