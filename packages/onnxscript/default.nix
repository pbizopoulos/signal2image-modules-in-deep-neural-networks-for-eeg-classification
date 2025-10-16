{
  pkgs ? import <nixpkgs> { },
}:
let
  onnx-ir = pkgs.python312Packages.buildPythonPackage rec {
    format = "wheel";
    pname = "onnx_ir";
    propagatedBuildInputs = [
      pkgs.python312Packages.ml-dtypes
      pkgs.python312Packages.onnx
    ];
    pythonImportsCheck = [ pname ];
    src = pkgs.python312Packages.fetchPypi rec {
      inherit pname version format;
      dist = python;
      python = "py3";
      sha256 = "KsT3DfLzXUHiTSWAxatJj4U/0B/U83JUhYo3mK/aIOk=";
    };
    version = "0.1.0";
  };
in
pkgs.python312Packages.buildPythonPackage rec {
  format = "wheel";
  pname = builtins.baseNameOf ./.;
  propagatedBuildInputs = [
    onnx-ir
    pkgs.python312Packages.packaging
    pkgs.python312Packages.typing-extensions
  ];
  pythonImportsCheck = [ pname ];
  src = pkgs.python312Packages.fetchPypi rec {
    inherit pname version format;
    dist = python;
    python = "py3";
    sha256 = "Igu+zMriKKKFvThRWcR6CfADxiPJY2r9VsyR31Bwaog=";
  };
  version = "0.3.2";
}
