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
        packagesAll = [
          pkgs.http-server
          pkgs.openssl
        ];
      in
      {
        devShells.all = pkgs.mkShell {
          buildInputs = packagesAll;
          shellHook = ''
            [ ! -z $STAGE ] && openssl req -subj '/C=..' -nodes -x509 -keyout tmp/privkey.pem -out tmp/fullchain.pem && http-server --tls --cert tmp/fullchain.pem --key tmp/privkey.pem || true
            exit    
          '';
        };
        devShells.check = pkgs.mkShell {
          buildInputs = packagesAll ++ [
            pkgs.biome
            pkgs.git
            pkgs.nixfmt-rfc-style
            pkgs.nodePackages.js-beautify
            pkgs.nodejs
          ];
          shellHook = ''
            set -e
            nix flake check
            nix fmt
            js-beautify --end-with-newline --indent-inner-html --no-preserve-newlines --type html --replace index.html
            [ -e script.js ] && biome check --unsafe --write script.js || true
            ls -ap | grep -v -E -x './|../|.env|.gitignore|CNAME|Makefile|index.html|flake.lock|flake.nix|prm/|pyscript/|python/|script.js|style.css|tmp/' | grep -q . && exit 1 || true
            test $(basename $(pwd)) = 'docs'
            exit    
          '';
        };
        devShells.default = pkgs.mkShell { buildInputs = packagesAll; };
        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
