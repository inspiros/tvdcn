#! /bin/bash

set -e
set -x

# Install CUDA 12.1: (requires thrust)
echo "Installing CUDA 12.1"
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe
./cuda.exe -s nvcc_12.1 cudart_12.1 thrust_12.1 cublas_dev_12.1 curand_dev_12.1 cusolver_dev_12.1 cusparse_dev_12.1
rm cuda.exe

export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin":$PATH
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"

# test
nvcc -V
