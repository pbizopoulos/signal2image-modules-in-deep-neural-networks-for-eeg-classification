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
        packages = with pkgs; [ ];
      in
      with pkgs;
      {
        devShells.default = pkgs.mkShell {
          buildInputs = packages ++ [
            http-server
            openssl
          ];
          shellHook = '''';
        };
        devShells.check = pkgs.mkShell {
          buildInputs = packages ++ [
            biome
            nixfmt-rfc-style
            nodePackages.js-beautify
            nodejs
          ];
        };
      }
    );
}
