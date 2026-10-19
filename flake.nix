{
  inputs.afairesi.url = "github:afairesi/afairesi";
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
