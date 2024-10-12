{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    check-python-script = {
      url = "github:pbizopoulos/check-python-script?dir=python";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixgl = {
      url = "github:nix-community/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {
    self,
    nixpkgs,
    flake-parts,
    check-python-script,
    nixgl,
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
          overlays = [nixgl.overlay];
        };
        dependencies = [
          onnxscript
          pkgs.nixgl.auto.nixGLDefault
          pkgs.python311Packages.matplotlib
          pkgs.python311Packages.onnx
          pkgs.python311Packages.pandas
          pkgs.python311Packages.scipy
          pkgs.python311Packages.torch-bin
          pkgs.python311Packages.torchvision-bin
          pkgs.python311Packages.types-requests
        ];
        onnxscript = pkgs.python311Packages.buildPythonPackage rec {
          pname = "onnxscript";
          version = "0.1.0.dev20240728";
          format = "wheel";
          src = pkgs.python311Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "Y8Cx4BQiWPf8QGAgnZMCQiWRQK5pqG/rHpwEHklEL78=";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [pkgs.python311Packages.ml-dtypes pkgs.python311Packages.packaging pkgs.python311Packages.onnx];
        };
      in {
        devShells = {
          all = pkgs.mkShell {
            PYTHONDONTWRITEBYTECODE = true;
            buildInputs = dependencies;
            shellHook = ''
              set -e
              nixGL python3 main.py || exit
              exit
            '';
          };
          check = pkgs.mkShell {
            buildInputs =
              dependencies
              ++ [
                check-python-script.packages.${system}.default
                pkgs.git
                pkgs.py-spy
                pkgs.python311Packages.coverage
                pkgs.python311Packages.mypy
                pkgs.ruff
              ];
            shellHook = ''
              set -e
              nix flake check --impure
              nix fmt
              check-python-script main.py
              ruff format --cache-dir tmp/ruff main.py
              ruff check --cache-dir tmp/ruff --exit-non-zero-on-fix --fix --select ALL --unsafe-fixes main.py
              mypy --cache-dir tmp/mypy --ignore-missing-imports --strict main.py
              [ -z $STAGE ] || (unset STAGE && coverage run --data-file=tmp/.coverage main.py && coverage html --data-file=tmp/.coverage --directory tmp/ --ignore-errors && py-spy record --format speedscope --output tmp/speedscope -- python main.py)
              ls -ap | grep -v -E -x './|../|.env|.gitignore|Makefile|flake.lock|flake.nix|main.py|prm/|pyproject.toml|python/|result|static/|templates/|tmp/' | grep -q . && exit 1
              test $(basename $(pwd)) = 'python'
              exit
            '';
          };
          default = pkgs.mkShell {buildInputs = dependencies;};
        };
        formatter = pkgs.alejandra;
      };
    };
}
