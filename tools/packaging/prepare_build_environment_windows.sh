#! /bin/bash

set -e
set -x

# Install CUDA 12.4: (requires thrust)
echo "Installing CUDA 12.4"
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.4.0/network_installers/cuda_12.4.0_windows_network.exe
./cuda.exe -s nvcc_12.4 cudart_12.4 thrust_12.4 cublas_dev_12.4 curand_dev_12.4 cusolver_dev_12.4 cusparse_dev_12.4
rm cuda.exe

export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin":$PATH
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"

# test
nvcc -V
