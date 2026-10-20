{
  inputs.afairesi = {
    inputs.nixpkgs.url = "github:NixOS/nixpkgs/fe13242281a67641c0102cdb1087f0ebe586c7df";
    url = "github:afairesi/afairesi";
  };
  outputs =
    inputs:
    inputs.afairesi.lib.mkFlake {
      inherit inputs;
      nixpkgs.config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
}
