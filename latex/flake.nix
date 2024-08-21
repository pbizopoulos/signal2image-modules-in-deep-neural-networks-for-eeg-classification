{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = {
    self,
    nixpkgs,
    flake-parts,
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {inherit system;};
        packages = with pkgs; [texlive.combined.scheme-full];
      in {
        devShells.all = pkgs.mkShell {
          buildInputs = packages;
          shellHook = ''
            set -e
            latexmk -outdir=tmp/ -pdf ms.tex
            touch tmp/ms.bbl
            cp tmp/ms.bbl .
            tar cf tmp/tex.tar ms.bbl ms.bib ms.tex $(grep "^INPUT ./" tmp/ms.fls | uniq | cut -b 9-)
            rm ms.bbl
            exit
          '';
        };
        devShells.check = pkgs.mkShell {
          buildInputs =
            packages
            ++ [
              pkgs.git
            ];
          shellHook = ''
            set -e
            nix flake check
            nix fmt
            latexmk -outdir=tmp/ -pdf ms.tex
            checkcites tmp/ms.aux
            chktex ms.tex
            lacheck ms.tex
            ls -ap | grep -v -E -x './|../|.gitignore|Makefile|flake.lock|flake.nix|ms.bib|ms.tex|prm/|python/|tmp/' | grep -q . && exit 1 || true
            test $(basename $(pwd)) = 'latex'
            exit
          '';
        };
        devShells.default = pkgs.mkShell {buildInputs = packages;};
        formatter = pkgs.alejandra;
      };
    };
}
