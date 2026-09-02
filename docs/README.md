# Documentation

Loom weaves LLMs into RTL. It compiles a language-model checkpoint into FPGA
accelerator IP, and it drives that IP from the host.

- [getting-started.md](getting-started.md) - the dev shells, the packages and
  the first commands.
- [architecture.md](architecture.md) - the compiler, the SoC variants, the
  host and device split, and the runtime modules.
- [genip.md](genip.md) - the `loom-genip` compiler: the flags, the artifacts
  and the `loom.json` manifest.
- [models.md](models.md) - the model formats, the quantization, the model
  features and the size limits.
- [hardware.md](hardware.md) - the targets and boards, the memory map, the
  clocking and the transports.
- [flashing.md](flashing.md) - how to build a bitstream and how to write it
  and the weights to a board.
- [runtime.md](runtime.md) - `loom-cli`, the OpenAI-compatible server and the
  C, C++ and Python bindings.
- [testing.md](testing.md) - the Dart suite, the Zig suite, the simulator and
  the checks on a board.
- [debugging.md](debugging.md) - how to find the layer that is broken.
- [status.md](status.md) - what works today, and what does not.
- [glossary.md](glossary.md) - short definitions of the terms in these docs.

## Where to start

| You want to | Read |
| ----------- | ---- |
| Build the tools | [getting-started.md](getting-started.md) |
| Know how Loom works | [architecture.md](architecture.md), then [hardware.md](hardware.md) |
| Compile a model into RTL | [genip.md](genip.md) and [models.md](models.md) |
| Build for a different FPGA | [hardware.md](hardware.md#select-a-target) |
| Put a design on a board | [flashing.md](flashing.md) |
| Generate text from a flashed board | [runtime.md](runtime.md) |
| Fix a board that misbehaves | [debugging.md](debugging.md) |

## Conventions

The tree builds through the Nix flake, and these docs assume that you are in
the matching dev shell:

```
nix develop      # or: nix develop .#ip   for the compiler
nix develop .#rt #                        for the runtime
```

The docs call the compiler `loom-genip`, which is the name that the `loom-ip`
package installs. From a checkout, run `dart run bin/loom_genip.dart` in
`ip/`. They call the runtime binary `loom-cli`, which `zig build` puts at
`runtime/zig-out/bin/loom-cli`.
