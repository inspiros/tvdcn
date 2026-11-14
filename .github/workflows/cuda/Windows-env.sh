#!/bin/bash
export TVDCN_BUILD_CUDA=1
export TVDCN_CUDA_ARCH_LIST='7.5 8.0 8.6 9.0 10.0 12.0+PTX'
# Exports from Windows.sh do not survive between workflow steps.
export CUDA_HOME='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0'
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin:$PATH"
