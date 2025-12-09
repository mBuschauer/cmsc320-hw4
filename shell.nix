{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs.buildPackages; [
    python313
    python313Packages.numpy
    python313Packages.tkinter
    uv

    python313Packages.jupyter
    python313Packages.ipython
  ];

  shellHook =
    ''
      uv venv --allow-existing
      uv lock
      uv sync 
      source .venv/bin/activate

      # jupyter-lab --ip 0.0.0.0
    '';

}