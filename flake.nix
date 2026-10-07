{
  inputs.afairesi.url = "github:afairesi/afairesi";
  outputs =
    inputs:
    inputs.afairesi.blueprint {
      inherit inputs;
      nixpkgs.config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    }
    // {
      formatter = inputs.afairesi.lib.mkFormatter { inherit (inputs) self; };
    };
}
