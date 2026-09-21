{ inputs, pkgs, ... }:
let
  packageDrv = inputs.self.packages.${pkgs.stdenv.system}.${packageName};
  packageName = baseNameOf ./.;
  pythonEnv = packageDrv.python.withPackages (
    ps:
    packageDrv.propagatedBuildInputs
    ++ (packageDrv.buildInputs or [ ])
    ++ [
      ps.hypothesis
      ps.pytest
    ]
  );
in
pkgs.runCommand packageName
  {
    inherit (packageDrv) src;
    PACKAGE_E2E_EXECUTABLE = pkgs.lib.getExe packageDrv;
    nativeBuildInputs =
      (packageDrv.nativeBuildInputs or [ ]) ++ packageDrv.propagatedBuildInputs ++ [ pythonEnv ];
  }
  ''
    export src PACKAGE_E2E_EXECUTABLE
    export HOME="$(mktemp -d)"
    mkdir -p "$out" packages
    ln -s "$src" "packages/${packageName}"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    cd "$out"
    "${pythonEnv}/bin/python" - <<'PYTHON'
    import os
    import sys
    from hypothesis import Phase, settings
    settings.register_profile("explicit", phases=[Phase.explicit])
    settings.load_profile("explicit")
    import pytest
    sys.exit(pytest.main([
        "-p", "no:cacheprovider",
        "--import-mode=importlib",
        os.environ["src"] + "/test_main.py",
    ]))
    PYTHON
  ''
