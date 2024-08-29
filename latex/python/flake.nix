{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    check-python-script = {
      url = "github:pbizopoulos/check-python-script?dir=python";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    onnxscript.url = "github:pbizopoulos/nixpkgs?dir=onnxscript";
  };
  outputs = {
    self,
    nixpkgs,
    flake-parts,
    check-python-script,
    onnxscript,
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        dependencies = [
          onnxscript.packages.${system}.default
          pkgs.python311Packages.matplotlib
          pkgs.python311Packages.onnx
          pkgs.python311Packages.pandas
          pkgs.python311Packages.scipy
          pkgs.python311Packages.torch-bin
          pkgs.python311Packages.torchvision-bin
          pkgs.python311Packages.types-requests
        ];
      in {
        devShells.all = pkgs.mkShell {
          PYTHONDONTWRITEBYTECODE = true;
          buildInputs = dependencies;
          shellHook = ''
            set -e
            python3 main.py || exit
            exit
          '';
        };
        devShells.check = pkgs.mkShell {
          buildInputs =
            dependencies
            ++ [
              check-python-script.packages.${system}.default
              pkgs.djlint
              pkgs.git
              pkgs.mypy
              pkgs.python3Packages.coverage
              pkgs.ruff
            ];
          shellHook = ''
            set -e
            nix flake check
            nix fmt
            check-python-script main.py
            ruff format --cache-dir tmp/ruff main.py
            ruff check --cache-dir tmp/ruff --exit-non-zero-on-fix --fix --select ALL --unsafe-fixes main.py
            mypy --cache-dir tmp/mypy --ignore-missing-imports --strict main.py
            [ -z $STAGE ] || (unset STAGE && coverage run --data-file=tmp/.coverage main.py && coverage html --data-file=tmp/.coverage --directory tmp/ --ignore-errors)
            if [ -d 'templates/' ]; then djlint templates/ --lint --profile=jinja --quiet --reformat; fi
            ls -ap | grep -v -E -x './|../|.env|.gitignore|Makefile|flake.lock|flake.nix|main.py|prm/|pyproject.toml|python/|result|static/|templates/|tmp/' | grep -q . && exit 1
            test $(basename $(pwd)) = 'python'
            exit
          '';
        };
        devShells.default = pkgs.mkShell {buildInputs = dependencies;};
        formatter = pkgs.alejandra;
      };
    };
}
