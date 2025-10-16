{ pkgs }:
pkgs.writeShellApplication {
  name = builtins.baseNameOf ./.;
  runtimeInputs = [ pkgs.texlive.combined.scheme-full ];
  text = ''
    REPOSITORY_DIR=$(git rev-parse --show-toplevel)
    REPO_NAME=$(realpath --relative-to="$HOME" "$REPOSITORY_DIR")
    latexmk -outdir="$HOME/$REPO_NAME/packages/latex/tmp/" -pdf ms.tex
  '';
}
