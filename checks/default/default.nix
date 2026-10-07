{ inputs, pkgs, ... }:
(inputs.afairesi or inputs.self).lib.mkPythonCheck {
  inherit pkgs;
  packageDrv = inputs.self.packages.${pkgs.stdenv.system}.${baseNameOf ./.};
  packageName = baseNameOf ./.;
}
