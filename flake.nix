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
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [cowsay lolcat];
          env.LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [];
        };
        apps.default = {
          type = "app";
          program = "${pkgs.writers.writeBashBin "stopp-rasisme" ''echo "rasisme er ikke greit!" | ${pkgs.cowsay}/bin/cowsay | ${pkgs.lolcat}/bin/lolcat --animate --duration 16''}/bin/stopp-rasisme";
        };
      }
    );
}
