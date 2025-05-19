#! /bin/bash

set -e
set -x

# Install CUDA 12.8: (requires thrust)
echo "Installing CUDA 12.8"
curl -L -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe
./cuda.exe -s nvcc_12.8 cudart_12.8 thrust_12.8 cublas_dev_12.8 curand_dev_12.8 cusolver_dev_12.8 cusparse_dev_12.8
rm cuda.exe

export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin":$PATH
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

# test
nvcc -V
