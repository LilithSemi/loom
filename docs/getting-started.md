# Getting started

Loom builds through its Nix flake. It supports `x86_64-linux` and
`aarch64-linux`. The dev shells give you Dart, Zig and the Python environment,
so the only prerequisite on the host is Nix with flakes.

## Dev shells

```
nix develop        # the compiler side. The same as nix develop .#ip
nix develop .#ip   # the Dart SDK and yq
nix develop .#rt   # Zig and a python env with nanobind, numpy, pytest, torch and transformers
```

The `ip` and `runtime` directories each hold an `.envrc` for direnv users.

## Build the packages

```
nix build .#loom-ip   # the compiler. It installs bin/loom-genip
nix build .#loom-rt   # the runtime: loom-cli and the C, C++ and Python bindings
```

`buildDartApplication` builds `loom-ip`, and its check phase runs the Dart
test suite. So `nix build` also tests the package.

`loom-rt` builds the Zig runtime with the `c`, `c++` and `python` bindings.
Select a subset with `-Dbindings=...`. See
[runtime.md](runtime.md#bindings).

The flake also gives an overlay at `overlays.default`. It adds `loom-ip` and
`loom-rt` to a nixpkgs `pkgs`.

## Work in the compiler (`ip/`, Dart)

```
cd ip
dart pub get                 # fetch the dependencies. Harbor comes from git
dart test                    # the full test suite
dart run bin/loom_genip.dart --help
```

The package name is `loom`. It uses ROHD, `rohd_hcl`, `rohd_bridge` and
Harbor. Harbor is the SoC framework. It gives the Wishbone fabric, the SRAM,
DDR3 and flash controllers, and the board catalog. It comes from
`git.lilithsemi.com/LilithSemi/harbor`.

See [genip.md](genip.md) for the compiler CLI.

## Work in the runtime (`runtime/`, Zig)

```
cd runtime
zig build test               # the unit tests, the C binding and the python pytest suite
zig build                    # loom-cli and the enabled bindings
./zig-out/bin/loom-cli help
```

A plain `zig build` builds the `c` and `c++` bindings. The Python binding
needs a Python environment with nanobind. The shell `nix develop .#rt` gives
one, and `nix build .#loom-rt` builds the binding.

## Format the tree

```
nix fmt
```

The tree uses [treefmt](https://github.com/numtide/treefmt): `dart-format` for
Dart, `nixfmt` for Nix and `jsonfmt` for JSON. The configuration is in
`treefmt.nix`.

## Next steps

- Compile a model into RTL: [genip.md](genip.md).
- Build and write the bitstream: [flashing.md](flashing.md).
- Speak to a board: [runtime.md](runtime.md).
- Learn what you are building: [architecture.md](architecture.md).
