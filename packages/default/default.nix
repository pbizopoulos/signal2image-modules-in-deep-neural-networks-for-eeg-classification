{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta.description = "A Python package.";
  nativeBuildInputs = [ pkgs.texliveFull ];
  propagatedBuildInputs = [
    python.pkgs.jinja2
    python.pkgs.matplotlib
    python.pkgs.numpy
    python.pkgs.pandas
    python.pkgs.pillow
    python.pkgs.scipy
    python.pkgs.torch
    python.pkgs.torchvision
  ];
  src = ./.;
  version = "0.0.0";
}
