# The runtime: `loom-cli`, the server and the bindings

The Zig runtime lives in `runtime/`. It has three faces: the `loom-cli`
binary, the `loom` library with its C, C++ and Python bindings, and the
OpenAI-compatible HTTP server inside the CLI.

Build it with `zig build` in a checkout. The binary lands at
`zig-out/bin/loom-cli`. The flake builds the same thing with `nix build
.#loom-rt`.

## `loom-cli` reference

```
loom-cli [-t uart|usb] [-p PATH] <command>
```

| Command | What it does |
| ------- | ------------ |
| `version` | Reads the VERSION CSR at bus address `0x000` and checks the `'LOOM'` magic. This address holds the `overlay` accelerator. |
| `info` | Reads VERSION and the geometry CSRs at the same address: the model name, the rows, the columns and the maximum tile. |
| `probe` | Tests the transport of an `fp` bitstream. It hammers single and burst register reads, it writes and reads back a CSR, and it re-reads a flash window. Run it first when generation misbehaves. |
| `matmul` | Runs a fixed 2x4 int8 matmul on an `overlay` board and compares it against the golden result. |
| `generate` | Generates text from a `loom-genip` model directory. Every transformer matmul runs on the device. |
| `serve` | Serves an OpenAI-compatible HTTP API. |
| `fetch` | Downloads a HuggingFace model repository. It needs no device. |

`version`, `info` and `matmul` read bus address `0x000`. The `stream` and `fp`
SoCs put their CSRs at `0x10000`, so use `probe` and `generate` with those
bitstreams. See [debugging.md](debugging.md).

Options:

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `-t`, `--transport uart\|usb` | `uart` | The host transport. |
| `-p`, `--port PATH` | `/dev/ttyACM0` | The serial device for the UART. |
| `-b`, `--baud RATE` | `1500000` | The UART baud rate. It must agree with the bitstream. |
| `-m`, `--model DIR` | none | The `loom-genip` output directory. |
| `-P`, `--prompt TEXT` | a bare BOS token | The prompt to continue. |
| `-n`, `--tokens N` | the model context | The maximum number of new tokens. |
| `--rep-window N` | `8` | Stop when the last N tokens repeat. `0` turns the check off. |
| `-l`, `--listen PORT` | `8080` | The port for `serve`. |
| `--stats` | off | Print the round trips and the write, poll and read times of each token. |
| `--poll-timeout N` | `1000000` | The maximum number of STATUS reads for one matmul. |
| `--poll-delay-us N` | `0` | The delay in microseconds between two STATUS reads. |
| `--repo`, `--out`, `--hf-token` | none | The options of `fetch`: the repository as `owner/name`, the destination directory, and a token for a gated repository. |

### `fetch`

```
loom-cli fetch --repo karpathy/tinyllamas --out model
```

`fetch` downloads `config.json`, `tokenizer.json`, `generation_config.json`
and the weights over HTTPS. For a sharded checkpoint it follows
`model.safetensors.index.json`. For a single-file checkpoint it takes
`model.safetensors`. Then compile the model:

```
loom-genip --soc fp --fp-flash --model model -o out
```

### `generate`

```
loom-cli generate --model out --prompt "Once upon a time" --stats
```

`generate` does this:

1. It loads the manifest, the tokenizer and the weight images from the model
   directory.
2. It reads the fp VERSION CSR and checks the magic.
3. It writes the hot weights into the BRAM cache, if the model has one.
4. It encodes the prompt and primes the KV cache. With no prompt it starts
   from the BOS token.
5. It streams each token as the token lands.

A token can take seconds over the UART, so the output is live. Generation
stops on the stop token of the model, on the repeat window, or on the token
cap. With no `-n`, the cap is the trained context of the model.

### `serve`

```
loom-cli serve --model out --listen 8080
```

The server gives an OpenAI-compatible API, so an agent or a chat client can
speak to the board:

- `GET /v1/models` gives the model list. The id comes from the silicon at
  start-up.
- `POST /v1/chat/completions` gives chat completions, as a stream or as one
  response.
- `GET /` gives a liveness reply.

The server takes one request for each connection and then closes it. Point any
OpenAI-compatible client at `http://localhost:8080/v1`.

## The `loom` library

`runtime/lib/loom.zig` is the library root. See
[architecture.md](architecture.md#the-runtime-runtime) for the module map. A
program opens a transport, takes its `Device` interface, loads a
`model.Config` from a `loom-genip` directory, and calls `forward.generate`.
The CLI takes the same path. `sim.zig` gives an in-process device, so the
whole stack also runs with no hardware.

## Bindings

The runtime exposes a C ABI in `bindings/c/`, and the C++ and Python layers
sit on it. Select the bindings with `zig build -Dbindings=c,c++,python`. A
plain `zig build` builds the C and C++ bindings. The Nix package builds all
three. The Python binding needs nanobind, and the shell `nix develop .#rt`
supplies it.

### C (`include/loom.h`)

Each call that can fail returns a `loom_status_t` and fills a `loom_error_t`
that the caller owns. There is no global error state. The lifecycle is:

```c
loom_runtime_create(&rt, err);
loom_device_open_uart(rt, "/dev/ttyACM0", 1500000, &dev, err);  // or open_usb / open_sim
loom_model_load(rt, "modeldir", &m, err);
loom_device_prepare(m, dev, err);
loom_generate(m, dev, &opts, token_cb, user_data, &produced, err);
```

The header also gives the parts that host glue needs: `loom_eval` and
`loom_mtp_draft` for contexts and speculative decoding, `loom_linear` and
`loom_linear_col_tiled` for single matmuls, `loom_model_matrix_by_name` for
matrix inspection, `loom_encode_image`, `loom_load_image` and `loom_eval_vlm`
for vision, the tokenizer calls, and the host ops such as `loom_rmsnorm`,
`loom_silu`, `loom_softmax`, `loom_rope` and `loom_moe_route`. The header
holds the full contract in its comments.

### C++ (`include/loom-cpp.hpp`)

A thin RAII wrapper over the same ABI. It throws in place of returning a
status. `bindings/c++/test/smoke.cpp` proves that the header compiles, links
and throws. `zig build test` runs it.

### Python (package `loom`)

A [nanobind](https://github.com/wjakob/nanobind) extension called `pyloom.so`,
which the `loom` package re-exports. It gives `Runtime`, `Device`, `Model`,
`Context`, `Matrix` and `LoomError`, and the host ops as numpy functions. The
names follow the C ABI.

```python
import numpy as np
import loom

loom.version()
loom.softmax(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
idx, w = loom.moe_route(logits, top_k=2, norm_topk=True)
```

The pytest suite in `bindings/python/tests/` runs as part of `zig build test`.
It covers the ops, the tokenizer, and generation through the simulator for
ternary, MTP and vision models.

The package also holds `loom.torch`, which needs `torch` and `transformers`.
It gives `LoomForCausalLM`, a `PreTrainedModel` over the bindings, so the
HuggingFace `generate()` method runs on the accelerator or on the simulator:

```python
from loom.torch import LoomForCausalLM
m = LoomForCausalLM.from_flashed(model_dir, transport="sim")
```

`transport` also takes `"usb"` or a tuple `("uart", port, baud)`. The KV cache
lives on the device, and the model keeps its own context, so it generates one
sequence at a time.
