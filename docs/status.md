# Status

What Loom does today, and what it does not do yet. This is a snapshot, not a
promise. The project is at version 0.0.1, and it is research code.

## Working

- **The `fp` datapath.** The fp16 W4A8 linear engine runs on an OrangeCrab
  r0.2. It reads int4 weights from the config flash, from DDR3, or from the
  on-chip BRAM cache. See [architecture.md](architecture.md).
- **End-to-end generation.** A `stories260K`-class model runs from a prompt to
  streamed tokens. Every transformer matmul runs on the device.
- **The compiler.** `loom-genip` reads HuggingFace, GGUF and llama2.c
  checkpoints, and it emits the SoC RTL, the board files, the weight images and
  the manifest. See [genip.md](genip.md).
- **Retargeting.** The SoC builds for Lattice ECP5, Lattice iCE40 and Xilinx
  7-series parts. `--board` takes a board from the catalog, and `--target`
  with `--pin` takes any ECP5 or iCE40 part. See
  [hardware.md](hardware.md#select-a-target).
- **The tiered weight cache.** `--fp-bram-cache-kb` moves the largest matrices
  into on-chip BRAM. The rest stay in the main store.
- **The host runtime.** `loom-cli` gives you `generate`, `serve`, `fetch` and
  the bring-up commands. The server speaks an OpenAI-compatible API. See
  [runtime.md](runtime.md).
- **Bindings.** C, C++ and Python bindings wrap one C ABI. The Python package
  also gives a HuggingFace `PreTrainedModel` wrapper.
- **The simulator.** `sim.zig` runs the full generate path with no hardware.
  The binding tests use it.
- **Model features.** GQA, gated MLPs, tied embeddings, MoE routing, MTP
  drafting, vision towers with a projector, and BitNet ternary weights. See
  [models.md](models.md).

## In progress

- **The `overlay` and `stream` SoCs.** These are bring-up steps. `overlay`
  proves the transport and the PE array. `stream` proves memory-backed
  matmuls. Neither runs a model.
- **The DDR3 weight store.** The controller works, but you must sweep
  `--ddr-read-tap` on the board to centre the read eye. Only the OrangeCrab
  entry is proven. The Arty S7-50 entry builds with the Xilinx PHY, but nobody
  has run it.
- **The full-forward datapath.** `ip/lib/src/hw/forward.dart` and the fp
  functional blocks (`fp_rmsnorm`, `fp_silu`, `fp_softmax`, `fp_rope`) move
  more of the model onto the device. They are exploratory, and the demo
  datapath does not use them.
- **The research tooling.** GPTQ, AWQ and SVD helpers in
  `ip/lib/src/golden/` are experimental.

## Not done yet

- **Host work stays on the host.** The device does the linear algebra only.
  The host does the embedding lookup, the norms, RoPE, softmax, the
  activations, the sampling and the MoE routing.
- **One sequence at a time.** There is no batching. The KV cache lives on the
  host, and the server closes the connection after each request.
- **USB with a model.** The USB transport works with the `overlay` SoC only.
  The `stream` and `fp` SoCs use the UART.
- **Large models.** The reference board holds approximately 14 MiB of weights
  in flash. Bigger models need DDR3 or a bigger part. See
  [models.md](models.md#choosing-a-model).
- **Boards other than the reference board.** Only `orangecrab-25f` runs today.
  The catalog also holds `ulx3s-85f` and `arty-s7-50`, but nobody has proven a
  Loom build on them. The `overlay` board shims still hold OrangeCrab pins, and
  `ulx3s-85f` has no DDR entry.
- **Speed.** The UART is the bottleneck. A token can take seconds. Use
  `--stats` to see where the time goes.
