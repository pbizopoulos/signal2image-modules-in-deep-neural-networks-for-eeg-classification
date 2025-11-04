{
  pkgs ? import <nixpkgs> { },
}:
let
  onnx-ir = pkgs.python312Packages.buildPythonPackage rec {
    format = "wheel";
    pname = "onnx_ir";
    propagatedBuildInputs = [
      pkgs.python312Packages.ml-dtypes
      pkgs.python312Packages.onnx
    ];
    pythonImportsCheck = [ pname ];
    src = pkgs.python312Packages.fetchPypi rec {
      inherit pname version format;
      dist = python;
      python = "py3";
      sha256 = "F/hvr4pTuXlDC94bxgIsehYrDRU0VQ3bF6HTfrmT52U=";
    };
    version = "0.1.12";
  };
  onnxscript = pkgs.python312Packages.buildPythonPackage rec {
    format = "wheel";
    pname = "onnxscript";
    propagatedBuildInputs = [
      onnx-ir
      pkgs.python312Packages.packaging
      pkgs.python312Packages.typing-extensions
    ];
    pythonImportsCheck = [ pname ];
    src = pkgs.python312Packages.fetchPypi rec {
      inherit pname version format;
      dist = python;
      python = "py3";
      sha256 = "sMM1X+o+7KuMopHai3ev3cqs062l7lkpQ5CgSeoSOTg=";
    };
    version = "0.5.6";
  };
  pythonEnv = pkgs.python312.withPackages (_ps: [
    onnxscript
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.onnx
    pkgs.python312Packages.pandas
    pkgs.python312Packages.scipy
    pkgs.python312Packages.torch-bin
    pkgs.python312Packages.torchvision-bin
  ]);
in
pkgs.stdenv.mkDerivation rec {
  buildInputs = [
    pkgs.texlive.combined.scheme-full
    pythonEnv
  ];
  installPhase = ''
    mkdir -p $out/bin
    echo '#!/usr/bin/env bash
      set -e
      package_dir=$HOME/github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/packages/default
      tmp_dir=$(mktemp -d)
      cp -r ${src}/* "$tmp_dir"
      cd "$tmp_dir"
      ${pythonEnv}/bin/python ./main.py
      ${pkgs.texlive.combined.scheme-full}/bin/latexmk -outdir=$package_dir/tmp -pdf ./ms.tex
      ' > $out/bin/${pname}
    chmod +x $out/bin/${pname}
  '';
  meta.mainProgram = pname;
  pname = builtins.baseNameOf src;
  src = ./.;
  version = "0.0.0";
}
