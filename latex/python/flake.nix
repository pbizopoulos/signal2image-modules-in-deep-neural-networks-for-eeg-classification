{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    check-python-script.url = "github:pbizopoulos/check-python-script/main?dir=python";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      check-python-script,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
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
            check-python-script.packages.${system}.default
            djlint
            mypy
            nixfmt-rfc-style
            ruff
          ];
        };
      }
    );
}
