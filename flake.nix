{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flakever.url = "github:numinit/flakever";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-parts,
      flakever,
      treefmt-nix,
      ...
    }@inputs:
    let
      flakeverConfig = flakever.lib.mkFlakever {
        inherit inputs;

        digits = [
          1
          2
          2
        ];
      };
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.flake-parts.flakeModules.easyOverlay
        inputs.treefmt-nix.flakeModule
      ];

      flake.versionTemplate = "1.1pre-<lastModifiedDate>-<rev>";

      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      perSystem =
        { system, pkgs, ... }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              self.overlays.default
            ];
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              dart-format.enable = true;
              nixfmt.enable = true;
              jsonfmt.enable = true;
            };
          };

          overlayAttrs = {
            flakever = flakeverConfig;
            loom-ip = pkgs.callPackage ./pkgs/loom-ip { };
            loom-rt = pkgs.callPackage ./pkgs/loom-rt { };
          };

          packages = {
            inherit (pkgs) loom-ip loom-rt;
          };

          devShells = {
            default = pkgs.loom-ip.shell;
            ip = pkgs.loom-ip.shell;
            rt = pkgs.loom-rt.shell;
          };
        };
    };
}
