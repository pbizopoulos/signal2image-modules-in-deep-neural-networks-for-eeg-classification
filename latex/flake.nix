{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        packages = with pkgs; [ texlive.combined.scheme-full ];
      in
      with pkgs;
      {
        devShells.default = pkgs.mkShell { buildInputs = packages; };
        devShells.check = pkgs.mkShell { buildInputs = packages ++ [ nixfmt-rfc-style ]; };
      }
    );
}
