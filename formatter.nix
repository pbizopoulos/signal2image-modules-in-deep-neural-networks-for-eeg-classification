# no-check
{ inputs, pkgs, ... }:
inputs.treefmt-nix.lib.mkWrapper pkgs {
  programs = {
    actionlint.enable = true;
    beautysh.enable = true;
    biome.enable = true;
    deadnix.enable = true;
    nixfmt.enable = true;
    prettier.enable = true;
    ruff-check.enable = true;
    ruff-format.enable = true;
    shellcheck.enable = true;
    shfmt.enable = true;
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
          "--cache-dir"
          "tmp/mypy"
          "--explicit-package-bases"
          "--ignore-missing-imports"
          "--strict"
        ];
        priority = 2;
      };
      nixfmt = {
        priority = 1;
        strict = true;
      };
      ruff-check = {
        options = [
          "--cache-dir"
          "tmp/ruff"
          "--select"
          "ALL"
          "--unsafe-fixes"
        ];
        priority = 2;
      };
      ruff-format = {
        options = [
          "--cache-dir"
          "tmp/ruff"
        ];
        priority = 1;
      };
      shfmt.options = [
        "--posix"
        "--simplify"
      ];
      statix.priority = 2;
      texfmt.options = [ "--nowrap" ];
    };
    global.excludes = [
      "*/prm/**"
      "*/tmp/**"
    ];
  };
}
