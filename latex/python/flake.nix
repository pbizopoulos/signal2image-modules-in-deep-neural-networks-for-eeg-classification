{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
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
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        check-python-script = pkgs.python3Packages.buildPythonPackage rec {
          name = "check-python-script";
          version = "0.0";
          format = "pyproject";
          src = fetchTarball rec {
            url = "https://api.github.com/repos/pbizopoulos/check-python-script/tarball/main#subdirectory=python";
            sha256 = "1rpzxyrkp8y86m7zryrvbhzwzifrij6y18gyhlkpgav7lp7rwxrk";
          };
          preBuild = ''
            cd python/
          '';
          propagatedBuildInputs = [
            pkgs.python3Packages.fire
            pkgs.python3Packages.libcst
            pkgs.python3Packages.setuptools
          ];
        };
        onnxscript = pkgs.python3Packages.buildPythonPackage rec {
          pname = "onnxscript";
          version = "0.1.0.dev20240701";
          format = "wheel";
          src = pkgs.python3Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "hTNedOxupvqtXwx3nu43Gb/kjeAPODBuTtKe4FP20qQ=";
            dist = python;
            python = "py3";
          };
        };
        packages = with pkgs; [
          onnxscript
          python3Packages.matplotlib
          python3Packages.onnx
          python3Packages.pandas
          python3Packages.scipy
          python3Packages.torch-bin
          python3Packages.torchvision-bin
          python3Packages.types-requests
        ];
      in
      with pkgs;
      {
        devShells.default = pkgs.mkShell { buildInputs = packages; };
        devShells.check = pkgs.mkShell {
          buildInputs = packages ++ [
            check-python-script
            djlint
            mypy
            nixfmt-rfc-style
            ruff
          ];
        };
      }
    );
}
