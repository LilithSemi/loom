# loom

Weave LLMs into RTL.

Loom is an end-to-end toolchain for running language models on FPGAs. It
compiles LLM checkpoints (HuggingFace `safetensors`, GGUF, llama2.c `.bin`)
into RTL for a Wishbone-SoC accelerator generated with
[ROHD](https://github.com/intel/rohd) and
[Harbor](https://git.lilithsemi.com/LilithSemi/harbor), targets an FPGA board
(the OrangeCrab ECP5 by default), and drives text generation from the host with
a Zig runtime that can also serve an OpenAI-compatible HTTP API.

The current demo datapath is a hybrid host/device split: the FPGA does the
linear algebra (the `fp` SoC's W4A8 engine - int4 weights streamed from config
flash / DDR3 / on-chip BRAM against int8 activations, int32 accumulation,
fp16 in/out), while the host orchestrates the nonlinear glue (embeddings,
RMSNorm/LayerNorm, RoPE, softmax, SiLU/GELU, sampling, MoE routing, vision
preprocessing). The target class is small models on small parts - think
llama2.c `stories260K` on an ECP5 25F, not a 70B on a datacenter card.

## Pipeline

1. **Fetch a model** (or point at a checkpoint you already have):

   ```
   loom-cli fetch --repo <owner>/<name> --out model
   ```

   downloads `config.json`, `tokenizer.json` and the safetensors weights
   (single-file or sharded) from HuggingFace. GGUF and llama2.c `.bin`
   checkpoints work too - see [docs/models.md](docs/models.md).

2. **Generate the IP:**

   ```
   loom-genip --soc fp --fp-flash --model model --tokenizer model/tokenizer.json -o out
   ```

   emits the SoC's SystemVerilog plus device tree, SVD, pin constraints and a
   synthesis Makefile, and the model artifacts the runtime consumes:
   `weights.bin`, `scales.bin`, `scales_flash.bin`, `glue.bin`, `loom.json`
   (model dims + matrix manifest + CSR/flash addresses) and `tokenizer.bin`.
   Details and all flags: [docs/genip.md](docs/genip.md).

3. **Build and flash** the bitstream and the weight image (`make` in the output
   dir, then `ecpprog`. On the OrangeCrab the weight image goes at
   `ecpprog -o 0x200000`). Steps and offsets:
   [docs/flashing.md](docs/flashing.md). Board and memory map:
   [docs/hardware.md](docs/hardware.md).

4. **Generate / serve** from the host:

   ```
   loom-cli generate --model out --prompt "Once upon a time"
   loom-cli serve --model out --listen 8080
   ```

   Every transformer matmul runs on the device; tokens stream back as they
   land. `serve` exposes `GET /v1/models` and `POST /v1/chat/completions`
   (streaming SSE or not). Full CLI reference:
   [docs/runtime.md](docs/runtime.md).

## Repository layout

| Path         | What lives there |
| ------------ | ------------------------------------------------------------------ |
| `ip/`        | `loom` Dart package: the model-to-RTL compiler. Frontends (HF config, GGUF, llama2.c), a typed model IR, weight loaders and the int4 flash-image packer, a golden reference model, and the ROHD/Harbor hardware library. The CLI tool `loom-genip`. |
| `runtime/`   | Zig host runtime: the `loom` library, the `loom-cli` binary (UART/USB transports, generation, OpenAI-compatible server, HF `fetch`), and C / C++ / Python bindings. |
| `pkgs/`      | Nix packages `loom-ip` and `loom-rt` that build the two above. |
| `flake.nix`  | flake-parts flake: packages, dev shells (`ip`, `rt`), treefmt formatting. |

The whole tree builds with Nix on `x86_64-linux` and `aarch64-linux`:

```
nix develop          # Dart toolchain (same as .#ip)
nix develop .#rt     # Zig toolchain + python env (nanobind, numpy, torch, ...)
nix build .#loom-ip .#loom-rt
```

Inside the shells: `dart test` (ip), `zig build test` (runtime), `dart run
bin/loom_genip.dart --help` / `zig-out/bin/loom-cli --help` for the tools.
Format the tree with `nix fmt`.

## Documentation

- [docs/README.md](docs/README.md) - the index of every page
- [docs/getting-started.md](docs/getting-started.md) - dev shells, packages, first commands
- [docs/architecture.md](docs/architecture.md) - the compiler, the SoC variants, the memory map
- [docs/genip.md](docs/genip.md) - the `loom-genip` compiler and the `loom.json` manifest
- [docs/models.md](docs/models.md) - the model formats, the quantization and the size limits
- [docs/hardware.md](docs/hardware.md) - the boards, the weight stores and the transports
- [docs/flashing.md](docs/flashing.md) - how to build a bitstream and write it to a board
- [docs/runtime.md](docs/runtime.md) - `loom-cli`, the server and the bindings
- [docs/testing.md](docs/testing.md) - the test suites, the simulator and the board checks
- [docs/debugging.md](docs/debugging.md) - how to find the layer that is broken
- [docs/status.md](docs/status.md) - what works today, and what does not
- [docs/glossary.md](docs/glossary.md) - short definitions of the terms

## Status

Early-stage research code (version 0.0.1). The `fp` SoC with flash-resident
weights has run `stories260K`-class models end to end on an OrangeCrab r0.2.
The `overlay` and `stream` SoCs are earlier bring-up steps, and parts of
`ip/lib/src/hw/` are exploratory. Expect rough edges. See
[docs/status.md](docs/status.md).

## License

Apache-2.0 - see [LICENSE](LICENSE).
