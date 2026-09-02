# Generating IP: `loom-genip`

`loom-genip` is the compiler. It builds a Harbor SoC from three choices:

- a **transport**, which is how the host speaks to the board,
- a **datapath**, which is the accelerator,
- a **target**, which is the board or the FPGA part.

It then writes synthesizable SystemVerilog and the board files: the device
tree, the SVD file, the `.lpf` constraints, `synth.tcl` and a Makefile. With
`--soc fp --model` it also writes the weight images, the tokenizer and the
manifest that the runtime reads.

The Nix package `loom-ip` installs the tool as `loom-genip`. From a checkout,
run `dart run bin/loom_genip.dart` instead.

## Options

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `-o`, `--output` | `out` | Output directory. |
| `--name` | per transport | The name of the SoC top module. |
| `--transport` | `usb` | `usb`, `uart`, or one of the link diagnostics `uartprobe`, `uartecho` and `uartechoswap`. See [debugging.md](debugging.md#diagnostic-bitstreams). |
| `--soc` | `overlay` | `overlay` is the 8x8 PE array. `stream` is the memory-backed int8 matmul. `fp` is the fp16 W4A8 linear engine that runs models. |
| `--board` | `orangecrab-25f` | A board from Harbor's catalog. It supplies the vendor, the device, the package, the oscillator, the pin sites and the program command. |
| `--target` | none | A part that the catalog does not hold, as `vendor:device:package`. For example `ecp5:25f:CSFBGA285` or `ice40:up5k:sg48`. It overrides `--board`. |
| `-p`, `--pin name=site` | OrangeCrab pinout | Add or replace one pin assignment, for example `uart_rx=M18`. Repeat the flag for more pins. |
| `--ddr` | off | Attach the DDR3 controller as a weight store. The OrangeCrab holds 128 MB. |
| `--fp-flash` | off | Keep the weights in the config SPI flash. The accelerator reads int4 weights directly from it. |
| `--fp-col-tiles` | `32` | The maximum number of inner-dimension tiles. The accelerator `maxCols` is two times this value. |
| `--fp-row-blocks` | `32` | The maximum number of output-row blocks. |
| `--fp-mhz` | `30` | The fabric clock in MHz, from the PLL. |
| `--fp-read-ahead` | `8` | The read-ahead length in words for the SPI flash. `1` disables it. |
| `--fp-bram-cache-kb` | `0` | The size of the on-chip hot-weight cache in KiB. `0` keeps all weights in the main store. |
| `--fp-baud` | `1500000` | The UART baud rate. It must agree with the runtime `--baud`. |
| `--model` | none | The model to compile: a HuggingFace directory, a GGUF file or a llama2.c `.bin` file. |
| `--tokenizer` | automatic | The tokenizer path. Loom compiles a `tokenizer.json` into `tokenizer.bin`, and copies a llama2.c `tok*.bin` without a change. |
| `--ddr-read-tap` | `40` | The static DDR3 read tap, from 0 to 127. Sweep it on the board to centre the read eye. |

The `--fp-*` flags apply to `--soc fp` only, with one exception: `--soc stream
--ddr` also reads `--fp-mhz`, because DDR3 does not close timing at the
oscillator frequency.

## Cookbook

```
# The overlay bring-up SoC with the USB transport:
loom-genip -o out

# The streaming int8 matmul SoC on the default board:
loom-genip --soc stream -o out

# The datapath that runs models, with the weights in config flash:
loom-genip --soc fp --fp-flash --model model -o out

# The same, with a 90 KiB hot-weight cache on chip:
loom-genip --soc fp --fp-flash --fp-bram-cache-kb 90 --model model -o out

# The weights in DDR3 in place of flash:
loom-genip --soc fp --ddr --model model -o out

# Another catalog board, or an arbitrary part:
loom-genip --soc fp --board arty-s7-50 --model model -o out
loom-genip --soc fp --target ice40:up5k:sg48 \
  --pin clk=35 --pin uart_tx=47 --pin uart_rx=46 --model model -o out
```

To build the bitstream from the output, see [flashing.md](flashing.md).

## Emitted model artifacts (`--soc fp --model`)

| File | Contents |
| ---- | -------- |
| `weights.bin` | The tile-major int4 weight image for the flash or DDR3 store. |
| `scales.bin` | The fp16 scales that match `weights.bin`, one per row or one per group. |
| `scales_flash.bin` | The resident scale image, with one fp16 in each 32-bit word. The accelerator reads it. |
| `weights_bram.bin`, `scales_bram.bin` | The images for the BRAM store. Loom writes them only with `--fp-bram-cache-kb`. |
| `glue.bin` | The fp16 host weights: the embedding table, the norm gammas, the q/k/v biases, the MoE routers and the vision tower weights. |
| `tokenizer.bin` | The runtime tokenizer, in LTB1 format or as a copied SentencePiece file. |
| `tokenizer_fixture.json` | Prompts and the expected token ids. It gates the runtime tokenizer. BPE models only. |
| `loom.json` | The manifest. |

`loom-genip` prints the size of each image. With a BRAM cache it also prints
the split of hot and cold matrices, and the bytes in each store.

## The `loom.json` manifest

The keys use snake case. The Zig runtime reads them in `model.Config`.

- **Dimensions**: `name`, `hidden`, `vocab`, `layers`, `num_heads`,
  `num_kv_heads`, `head_dim`, `intermediate`, `max_seq`, `rope_theta`,
  `norm_eps`, `tie_embeddings` and `max_cols`. `max_cols` is the column
  capacity of the accelerator. The runtime tiles a wider matrix.
- **Quantization**: `quant` holds `bitnet_ternary` for a ternary model. It is
  absent for other models.
- **Optional features**: `moe` gives the expert count and the top-k value.
  `mtp` gives the prediction modules. `vision` gives the ViT tower.
  `projector` gives the vision-language projector. `image_token_index` gives
  the token that an image replaces.
- **Addresses**: `csr_base`, `flash_weight_base` and `scale_flash_base`. A
  model with a cache also has `bram_cache_kb`, `bram_weight_base` and
  `bram_scale_base`.
- **Glue offsets**: `embed_offset`, `final_norm_offset` and `layer_glue`.
- **Matrices**: one entry for each matrix, with `name`, `store` (`bram` or
  `flash`), `weight_offset`, the scale offsets, `rows`, `cols`, `col_tiles`
  and `groups`.

Read the manifest when a matrix looks wrong. See
[debugging.md](debugging.md#check-the-manifest).
