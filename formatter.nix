{
  flake,
  inputs,
  pkgs,
  ...
}:
let
  formatter = treefmtEval.config.build.wrapper;
  treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs {
    programs = {
      actionlint.enable = true;
      beautysh.enable = true;
      biome.enable = true;
      deadnix.enable = true;
      nixfmt = {
        enable = true;
        strict = true;
      };
      prettier.enable = true;
      ruff-check = {
        enable = true;
        extendSelect = [ "ALL" ];
      };
      ruff-format.enable = true;
      shellcheck.enable = true;
      shfmt = {
        enable = true;
        simplify = true;
      };
      statix.enable = true;
      texfmt.enable = true;
      yamlfmt.enable = true;
    };
    projectRootFile = "flake.nix";
    settings = {
      formatter = {
        bibtex-tidy = {
          command = pkgs.bibtex-tidy;
          includes = [ "*.bib" ];
          options = [
            "--duplicates"
            "--no-align"
            "--no-wrap"
            "--sort"
            "--sort-fields"
            "--v2"
          ];
        };
        mypy = {
          command = pkgs.mypy;
          includes = [ "*.py" ];
          options = [
            "--explicit-package-bases"
            "--ignore-missing-imports"
            "--strict"
          ];
        };
        ruff-check.options = [ "--unsafe-fixes" ];
        shfmt.options = [ "--posix" ];
        texfmt.options = [ "--nowrap" ];
      };
      global.excludes = [
        "*/prm/**"
        "*/tmp/**"
      ];
    };
  };
in
formatter
// {
  passthru = formatter.passthru // {
    tests.check = treefmtEval.config.build.check flake;
  };
}
