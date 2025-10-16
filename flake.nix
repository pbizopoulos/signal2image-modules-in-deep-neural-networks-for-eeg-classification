{
  inputs = {
    canonicalization.url = "github:pbizopoulos/canonicalization";
    nixpkgs.follows = "canonicalization/nixpkgs";
  };
  outputs =
    inputs:
    inputs.canonicalization.blueprint {
      inherit inputs;
      nixpkgs.config.allowUnfree = true;
    }
    // {
      inherit (inputs.canonicalization) formatter;
    };
}
