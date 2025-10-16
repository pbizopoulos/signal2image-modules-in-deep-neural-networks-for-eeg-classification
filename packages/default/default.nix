{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
pkgs.writeShellApplication {
  name = builtins.baseNameOf ./.;
  runtimeInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.python
    pkgs.texlive.combined.scheme-full
  ];
  text = ''
    python
    cd ${../latex}
    latexmk -outdir="$HOME"/github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/packages/latex/tmp/ -pdf ms.tex
    xdg-open "$HOME"/github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/packages/latex/tmp/ms.pdf
  '';
}
