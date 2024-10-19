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
        dependencies = [
          pkgs.http-server
          pkgs.openssl
        ];
      in {
        devShells = {
          all = pkgs.mkShell {
            buildInputs = dependencies;
            shellHook = ''
              set -e
              if [ -n "$STAGE" ]; then
                openssl req -keyout tmp/privkey.pem -nodes -out tmp/fullchain.pem -subj '/C=..' -x509
                http-server --cert tmp/fullchain.pem --key tmp/privkey.pem --tls
                exit 1
              else
                exit 0
              fi
            '';
          };
          check = pkgs.mkShell {
            buildInputs =
              dependencies
              ++ [
                pkgs.biome
                pkgs.git
                pkgs.nodePackages.js-beautify
                pkgs.nodePackages.prettier
                pkgs.nodejs
              ];
            shellHook = ''
              set -e
              nix flake check
              nix fmt
              prettier --write .
              js-beautify --end-with-newline --indent-inner-html --no-preserve-newlines --type html --replace index.html
              [ -e script.js ] && biome check --unsafe --write script.js || true
              ls -ap | grep -v -E -x './|../|.env|.gitignore|CNAME|Makefile|index.html|flake.lock|flake.nix|prm/|pyscript/|python/|script.js|style.css|tmp/' | grep -q . && exit 1
              test $(basename $(pwd)) = 'docs'
              exit
            '';
          };
          default = pkgs.mkShell {buildInputs = dependencies;};
        };
        formatter = pkgs.alejandra;
      };
    };
}
