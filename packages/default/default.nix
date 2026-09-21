{ pkgs, ... }:
let
  nativeDeps = [ pkgs.texliveFull ];
  pname = baseNameOf ./.;
  python = pkgs.python3;
  pythonDeps = [
    python.pkgs.jinja2
    python.pkgs.matplotlib
    python.pkgs.numpy
    python.pkgs.pandas
    python.pkgs.pillow
    python.pkgs.scipy
    python.pkgs.torch
    python.pkgs.torchvision
  ];
  shellHook = "";
in
python.pkgs.buildPythonPackage {
  inherit pname;
  inherit shellHook;
  installPhase = ''
    install -Dm644 main.py "$out/${python.sitePackages}/$pname/__init__.py"
    mkdir -p "$out/bin"
    printf '%s\n' '#!${python.interpreter}' "from $pname import main" 'main()' > "$out/bin/$pname"
    chmod 755 "$out/bin/$pname"
    if [ -d prm ]; then
      cp -R prm/ "$out/${python.sitePackages}/$pname/"
    fi
  '';
  meta = {
    description = "A Python package.";
    mainProgram = pname;
  };
  nativeBuildInputs = nativeDeps;
  passthru.python = python;
  propagatedBuildInputs = pythonDeps;
  pyproject = false;
  src = ./.;
  strictDeps = true;
  version = "0.0.0";
}
