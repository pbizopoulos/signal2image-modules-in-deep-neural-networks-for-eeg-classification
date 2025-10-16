{ inputs, pkgs, ... }:
pkgs.testers.runNixOSTest rec {
  name = builtins.baseNameOf ./.;
  nodes.machine = {
    environment.systemPackages = [
      inputs.self.packages.${pkgs.stdenv.system}.${name}
    ];
    networking.useDHCP = true;
    virtualisation.memorySize = 8192;
  };
  testScript = ''machine.succeed("DEBUG=1 ${name}")'';
}
