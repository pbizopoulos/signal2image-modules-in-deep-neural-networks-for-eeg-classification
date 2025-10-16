{
  pkgs ? import <nixpkgs> { },
}:
pkgs.writeShellApplication {
  name = builtins.baseNameOf ./.;
  runtimeInputs = [ pkgs.nodePackages.http-server ];
  text = ''set +u && [ -z "$DEBUG" ] && http-server ${./.}'';
}
