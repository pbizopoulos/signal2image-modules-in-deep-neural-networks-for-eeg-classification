{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta = {
    description = "Signal2Image Modules in Deep Neural Networks for EEG Classification";
    homepage = "https://arxiv.org/abs/1904.13216";
    longDescription = ''
      Publication: https://ieeexplore.ieee.org/abstract/document/8856620/
    '';
  };
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
}
