{
  inputs.canonical.url = "github:pbizopoulos/canonical";
  outputs =
    inputs:
    inputs.canonical.blueprint {
      inherit inputs;
      nixpkgs.config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    }
    // {
      inherit (inputs.canonical) formatter;
    };
}
