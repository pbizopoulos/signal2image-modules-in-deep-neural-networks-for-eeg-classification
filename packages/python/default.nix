{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
pkgs.python312Packages.buildPythonPackage rec {
  installPhase = ''
    mkdir -p $out/bin
    cp ./main.py $out/bin/${pname}
    cp -r ./prm/ $out/bin/
  '';
  meta.mainProgram = pname;
  pname = builtins.baseNameOf src;
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.onnxscript
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.pandas
    pkgs.python312Packages.scipy
    pkgs.python312Packages.torch-bin
    pkgs.python312Packages.torchvision-bin
  ];
  pyproject = false;
  src = ./.;
  version = "0.0.0";
}
