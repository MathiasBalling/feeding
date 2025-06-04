{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "aarch64-darwin"; }; # Adjust if using x86_64
    in
    {
      devShells.aarch64-darwin.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          python312
          # python312Packages.pip
          # python312Packages.setuptools
          # python312Packages.wheel
          poetry
          bazel_5
          # libffi
        ];
        shellHook = ''
          if [ -f ./env/bin/activate ]; then
            source ./env/bin/activate
          fi
        '';
      };
    };
}
