{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    check-python-script.url = "github:pbizopoulos/check-python-script?dir=python";
    onnxscript.url = "github:pbizopoulos/nixpkgs?dir=onnxscript";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      check-python-script,
      onnxscript,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        packagesAll = [
          onnxscript.packages.${system}.default
          pkgs.python3Packages.matplotlib
          pkgs.python3Packages.onnx
          pkgs.python3Packages.pandas
          pkgs.python3Packages.scipy
          pkgs.python3Packages.torch-bin
          pkgs.python3Packages.torchvision-bin
          pkgs.python3Packages.types-requests
        ];
        packagesCheck = [
          check-python-script.packages.${system}.default
          pkgs.djlint
          pkgs.mypy
          pkgs.nixfmt-rfc-style
          pkgs.ruff
        ];
      in
      {
        devShells.all = pkgs.mkShell {
          buildInputs = packagesAll;
          shellHook = ''
            set -e
            python3 main.py || exit
            exit
          '';
        };
        devShells.check = pkgs.mkShell {
          buildInputs = packagesAll ++ packagesCheck;
          shellHook = ''
            set -e
            nix flake check
            nix fmt
            check-python-script main.py
            ruff format --cache-dir tmp/ruff main.py
            ruff check --cache-dir tmp/ruff --exit-non-zero-on-fix --fix --select ALL --unsafe-fixes main.py
            mypy --cache-dir tmp/mypy --ignore-missing-imports --strict main.py
            if [ -d 'templates/' ]; then djlint templates/ --lint --profile=jinja --quiet --reformat; fi
            [ -z $STAGE ] || (unset STAGE && pydoc -w main && mv main.html tmp/)
            ls -ap | grep -v -E -x './|../|.env|.gitignore|Makefile|flake.lock|flake.nix|main.py|prm/|pyproject.toml|python/|result|static/|templates/|tmp/' | grep -q . && exit 1 || true
            test $(basename $(pwd)) = 'python'
            exit
          '';
        };
        devShells.default = pkgs.mkShell { buildInputs = packagesAll; };
        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
