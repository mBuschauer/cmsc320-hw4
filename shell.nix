{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:
let nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.beta;
in pkgs.mkShell {
  buildInputs = with pkgs.buildPackages; [
    python313
    python313Packages.numpy
    python313Packages.tkinter
    uv

    procps
    gnumake
    util-linux
    m4
    gperf
    stdenv.cc.cc
    binutils
    cudatoolkit
    nvidiaPackage
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export EXTRA_LDFLAGS="-L/lib -L${nvidiaPackage}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

    uv venv --allow-existing
    uv lock
    uv sync 
    source .venv/bin/activate

    # jupyter-lab --ip 0.0.0.0
  '';

}
