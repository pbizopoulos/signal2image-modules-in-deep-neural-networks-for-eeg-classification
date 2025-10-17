{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
let
  pythonEnv = pkgs.python312.withPackages (_ps: [
    inputs.self.packages.${pkgs.stdenv.system}.onnxscript
    pkgs.python312Packages.matplotlib
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
