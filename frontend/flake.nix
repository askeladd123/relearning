{
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        # nodePkgs = import ./default.nix {inherit pkgs system;};
      in {
        devShell = pkgs.mkShell {
          # inputsFrom = [nodePkgs.shell];
          # buildInputs = with pkgs; [esbuild];
          buildInputs = with pkgs; [node2nix];
        };
      }
    );
}
