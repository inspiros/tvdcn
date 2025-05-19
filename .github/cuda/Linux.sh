#! /bin/bash

set -e
set -x

# Install lapack, blas
yum install -y lapack-devel blas-devel

# Install CUDA 12.8:
echo "Installing CUDA 12.8"
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-8-12.8.93-1 \
    cuda-cudart-devel-12-8-12.8.90-1 \
    libcublas-devel-12-8-12.8.4.1-1 \
    libcurand-devel-12-8-10.3.9.90-1 \
    libcusolver-devel-12-8-11.7.3.90-1 \
    libcusparse-devel-12-8-12.5.8.93-1
ln -s cuda-12.8 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
