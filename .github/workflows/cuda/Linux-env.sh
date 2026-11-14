#!/bin/bash
export TVDCN_BUILD_CUDA=1
# CUDA 13 supports sm_75 and newer; retain PTX for future GPUs.
export TVDCN_CUDA_ARCH_LIST='7.5 8.0 8.6 9.0 10.0 12.0+PTX'
