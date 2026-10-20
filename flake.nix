{
  inputs.afairesi = {
    inputs.nixpkgs.url = "github:NixOS/nixpkgs/fe13242281a67641c0102cdb1087f0ebe586c7df";
    url = "github:afairesi/afairesi";
  };
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
