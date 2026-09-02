# Testing

Loom has three test levels: the Dart suite in `ip/`, the Zig suite in
`runtime/`, and the checks you run on a board. The first two need no hardware.

## The compiler suite (`ip/`)

```
cd ip
dart pub get
dart test
```

`nix build .#loom-ip` runs the same suite in its check phase.

| Directory | What it covers |
| --------- | -------------- |
| `test/frontend/` | Config readers for HuggingFace, GGUF, Qwen2 biases and vision configs. |
| `test/ir/` | The model graph, the tensors and the graph validation. |
| `test/weights/` | The safetensors, sharded safetensors and GGUF loaders, and the int4 flash image. |
| `test/golden/` | The reference forward pass: ops, attention, quantization, MoE, MTP, vision and the metrics. |
| `test/hw/` | ROHD simulations of the hardware: the PE array, the matmul and linear engines, the fp blocks, the transports and the SoC elaboration. |
| `test/tokenizer/` | The BPE reader and the LTB1 writer. |
| `test/cli` and `test/nano` | The command-line helpers and the compact model description. |

The hardware tests run a ROHD simulation and compare the result against the
golden model. `test/hw/soc_gen_test.dart` elaborates the SoC variants, so a
break in the generator fails the suite before you reach synthesis.

## The runtime suite (`runtime/`)

```
cd runtime
zig build test
```

The step runs:

1. the Zig unit tests of the library modules (`quant`, `protocol`, `fp`,
   `model`, `sim`, `linear`, `ops` and more),
2. the C binding tests,
3. the C++ smoke test, which proves that the header compiles, links and
   throws,
4. the Python pytest suite, when you build the Python binding.

A plain `zig build` builds the C and C++ bindings. Add the Python binding with
`zig build -Dbindings=c,c++,python`. The Python binding needs nanobind, and the
shell `nix develop .#rt` supplies it.

The Python tests in `runtime/bindings/python/tests/` generate text through the
simulator for a ternary model, an MTP model and a vision model. They also test
the ops and the tokenizer.

## Testing without hardware

`runtime/lib/loom/sim.zig` is an in-process device. It answers the same
register and buffer operations as the silicon, so the model runner, the
bindings and the tests use one code path. Open it from C with
`loom_device_open_sim`, or from Python with `transport="sim"`.

The simulator does not model the transport. It shows logic errors, but it does
not show timing problems, flash-read problems or baud problems.

## The tokenizer gate

`loom-genip --model` writes `tokenizer_fixture.json` next to `tokenizer.bin`
when the model uses a BPE tokenizer. The file holds prompts and the token ids
that the compiler produced. The runtime encoder must give the same ids. Use it
when you change either tokenizer.

## Testing on a board

The commands split by SoC variant, because the accelerator sits at a different
bus address in each one.

| Bitstream | Commands | What they prove |
| --------- | -------- | --------------- |
| `overlay` | `loom-cli version`, `info`, `matmul` | The link is alive, and the PE array gives the golden result. These commands read the accelerator at bus address `0x000`. |
| `fp` | `loom-cli probe`, then `loom-cli generate` | `probe` reads the fp CSRs at `0x10000` and the weight window at `0x20200000`. `generate` runs the model. |

The `stream` SoC also puts its CSRs at `0x10000`, so `version`, `info` and
`matmul` do not work against a `stream` bitstream.

Run `probe` first when generation misbehaves. It hammers single and burst
register reads, it writes and reads back a CSR, and it re-reads one flash
window many times. A clean probe means that you can trust the link. Then look
at the compute path or at the weight image.

Use `--stats` on `generate` to see the round-trip count and the time of each
token. [debugging.md](debugging.md) tells you what each failure means.

## Formatting

```
nix fmt
```

The tree uses treefmt: `dart-format` for Dart, `nixfmt` for Nix and `jsonfmt`
for JSON.
