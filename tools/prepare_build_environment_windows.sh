#! /bin/bash

set -e
set -x

# Install CUDA 11.8:
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe
./cuda.exe -s nvcc_11.8 cudart_11.8 cublas_dev_11.8 curand_dev_11.8 cusolver_dev_11.8 cusparse_dev_11.8
rm cuda.exe
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"

# test
nvcc -V
