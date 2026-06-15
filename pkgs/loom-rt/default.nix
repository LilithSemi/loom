{
  lib,
  stdenv,
  flakever,
  mkShell,
  zig,
  python3,
  bindings ? [
    "c"
    "c++"
    "python"
  ],
}:
let
  needsPython = (lib.lists.findFirst (x: x == "python") false bindings) != false;

  pyenv = python3.withPackages (
    ps: with ps; [
      nanobind
      numpy
      pytest
      torch
      transformers
    ]
  );
in
stdenv.mkDerivation (finalAttrs: {
  pname = "loom-rt";
  inherit (flakever) version;

  src = ../../runtime;

  nativeBuildInputs = [
    zig
  ];

  buildInputs = lib.optional needsPython pyenv;

  zigBuildFlags = lib.map (n: "-Dbindings=${n}") bindings;

  passthru.shell = mkShell {
    name = "loom-rt-dev-shell";
    packages = [
      zig
    ]
    ++ lib.optional needsPython pyenv;
  };
})
