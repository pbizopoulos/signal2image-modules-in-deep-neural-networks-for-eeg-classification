{
  inputs.afairesi.url = "github:afairesi/afairesi";
  outputs =
    inputs:
    let
      flake = inputs.afairesi.lib.mkFlake {
        inherit inputs;
        nixpkgs.config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    flake
    // {
      packages = builtins.mapAttrs (
        _: packages: packages // { default = packages.generate_paper; }
      ) flake.packages;
    };
}
