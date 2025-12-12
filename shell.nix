{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

let
  nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.beta;
  gpu = "nvidia";
in pkgs.mkShell {
  buildInputs = with pkgs;
    [
      python313
      python313Packages.numpy
      python313Packages.tkinter
      tcl
      tk
      uv
      stdenv.cc.cc
    ] ++ (if gpu == "nvidia" then
      with pkgs; [
        procps
        gnumake
        util-linux
        m4
        gperf
        binutils
        cudatoolkit
        nvidiaPackage
      ]
    else if gpu == "amd" then
      with pkgs; [
        rocmPackages.clr
        rocmPackages.clr.icd
        rocmPackages.rocm-smi
        rocmPackages.rocminfo
      ]
    else
      [

      ]);

  GPU_TYPE = gpu;
  MPLBACKEND = "TkAgg";
  TCL_LIBRARY = "${pkgs.tcl}/lib/tcl8.6";
  TK_LIBRARY = "${pkgs.tk}/lib/tk8.6";


  shellHook = ''
    case "$GPU_TYPE" in
      nvidia)
        echo "Running NVIDIA-specific shellHook code"
        export CUDA_PATH=${pkgs.cudatoolkit}
        export EXTRA_LDFLAGS="-L/lib -L${nvidiaPackage}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        ;;
      amd)
        echo "Running AMD-specific shellHook code"
        export ROCM_PATH=${pkgs.rocmPackages.clr.icd}
        ;;
      *)
        echo "Unknown GPU, running fallback shellHook code"
        ;;
    esac

    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
    export HF_HOME="$PWD/data/HuggingFace"

    uv venv --allow-existing
    uv lock
    uv sync
    source ./.venv/bin/activate
  '';
}
