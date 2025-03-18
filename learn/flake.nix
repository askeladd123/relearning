{
  inputs.utils.url = "github:numtide/flake-utils";
  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "cool-program";
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [];
          env.LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [];
        };
        packages.default = pkgs.stdenv.mkDerivation {
          name = name;
          src = ./.;
          buildPhase = ''
            echo "#!/usr/bin/env bash" > ${name}
            echo "echo hello" >> ${name}
            chmod +x ${name}
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp ${name} $out/bin/
          '';
        };
        apps.${system}.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/${name}";
        };
      }
    );
}
