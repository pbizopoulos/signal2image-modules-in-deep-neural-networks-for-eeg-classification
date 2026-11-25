{
  inputs.afairesi.url = "github:afairesi/flake-conventions";
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
