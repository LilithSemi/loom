# Models: formats, features and scale

## Input formats

`--model` takes three kinds of checkpoint.
`ip/lib/src/frontend/model_source.dart` decides which one you gave it.

| Format | What it looks like | Notes |
| ------ | ----------------- | ----- |
| HuggingFace directory | `config.json`, `model.safetensors` or an index with shards, and `tokenizer.json` | The main path. `loom-cli fetch --repo owner/name --out dir` downloads one. Use `--hf-token` for a gated repository. |
| llama2.c `.bin` | One checkpoint file, with a `tok*.bin` SentencePiece tokenizer | The tiny-model path, such as `stories260K`. This class runs end to end on silicon. Loom copies the tokenizer without a change. |
| GGUF | One `.gguf` file | The metadata and the tokenizer come from the file itself. |

Loom compiles a HuggingFace BPE `tokenizer.json` into the LTB1 file
`tokenizer.bin`. It also writes `tokenizer_fixture.json`, which holds prompts
and the expected token ids. The runtime encoder must give the same ids. The
stories and SmolLM2 tokenizers use `<|endoftext|>` as the BOS and EOS token,
and they do not add a BOS token to a prompt.

## How Loom stores the weights

All transformer matrices go into the int4 tile-major weight image of the
device, with a parallel fp16 scale image. This includes the projections, the
MLP gate, up and down matrices, the LM head when the model does not tie it,
the MoE experts, the MTP modules and the vision matrices. Loom uses one scale
for each row, or one scale for each group when the source model quantized that
way. The `groups` key in the manifest says which.

The device then does W4A8 arithmetic: int4 weights against int8 activations,
into int32 accumulators. It gives fp16 back with `acc * rowScale * actScale`.

A matrix wider than `max_cols` goes into the image in column blocks. The
runtime tiles it across more than one call with `loom_linear_col_tiled`. The
value of `max_cols` comes from `--fp-col-tiles`.

`--fp-bram-cache-kb` splits the model across two stores. The largest matrices
by byte size fill the on-chip cache. The rest stay in the main store.

The host needs some weights at fp16, and those go into `glue.bin`: the
embedding table, the layer and final norm gammas, the Qwen2 q/k/v biases, the
MoE routers, and the norms, biases and embeddings of the vision tower.

## Features

- **Attention**: GQA, which is `num_kv_heads` below `num_heads`. Also a
  configurable `rope_theta`, tied embeddings and gated MLPs.
- **MoE**: the host runs the fp16 router and selects the top-k experts. It
  renormalizes the weights when the model asks for it. The expert matrices use
  the names `layers.$i.experts.$e.{gate,up,down}_proj`. A layer is an MoE
  layer when its `layer_glue` entry holds a `router` offset.
- **MTP**: extra prediction modules. Their `eh_proj` and block matrices go
  into the int4 image with the names `mtp.$m.*`. The device drafts the tokens,
  and the host verifies them with one full evaluation through
  `loom_mtp_draft`.
- **Vision and language**: a ViT tower with the `vision.*` matrices and host
  glue, and a projector with the `projector.*` matrices. The projector has one
  or two layers. Its `scale_factor` sets the Idefics3 pixel shuffle, and a
  value of 1 gives the LLaVA behaviour with no shuffle. The host decodes the
  JPEG image, applies `image_mean` and `image_std` from the manifest, runs the
  tower, and gives the embeddings to `loom_eval_vlm`.
- **BitNet ternary**: Loom finds a ternary checkpoint when it loads the model,
  and it writes `quant: bitnet_ternary` into the manifest. The processing
  elements then need no multiplier. This is a build-time choice, so the model
  must be known before you generate the SoC.
- **Research tools**: `ip/lib/src/golden/` holds GPTQ, AWQ and SVD helpers,
  and reference code for MoE, MTP and vision. Treat these as experimental.

## Choosing a model

Three properties of the target set the size of the model that it runs:

| Property | What it limits |
| -------- | -------------- |
| The size of the config flash, less the `0x200000` that the bitstream holds | The weights and the scales that stay resident on the board. |
| The BRAM of the part | The hot-weight cache from `--fp-bram-cache-kb`, and the size of the accelerator itself. |
| An external DRAM, if the board has one | The weight store for a model that flash cannot hold. |

The reference board is an OrangeCrab with an ECP5 25F. It gives approximately
14 MiB of flash for the weights, approximately 90 KiB of BRAM for the cache,
which is close to the full weight set of `stories260K` at approximately 130
KB, and an optional 128 MB DDR3 store. A larger part gives more of each. See
[hardware.md](hardware.md#select-a-target).

| Model size | What to do |
| ---------- | ---------- |
| `stories260K` class | Keep the weights in flash. Add `--fp-bram-cache-kb 90` to pull the hot matrices on chip. |
| A few MB | Flash still holds it, but the flash reads make generation slow. Move to `--ddr` for more speed. |
| More than the flash holds | Use `--ddr`, use a board with a larger flash, or use a smaller model. |

`loom-genip` checks the images against the 16 MiB flash of the reference
board. It gives a warning above 14 MiB of weights, and it stops with an error
when the weights and the scales together overrun 16 MiB.

`loom-genip` prints the weight split when you configure a BRAM cache, and it
writes the per-matrix manifest. Read the manifest to see where each matrix
landed.
