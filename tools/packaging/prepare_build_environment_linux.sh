#! /bin/bash

set -e
set -x

# Install lapack, blas
yum install -y lapack-devel blas-devel

# Install CUDA 12.1:
echo "Installing CUDA 12.1"
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-1-12.1.105-1 \
    cuda-cudart-devel-12-1-12.1.105-1 \
    libcublas-devel-12-1-12.1.3.1-1 \
    libcurand-devel-12-1-10.3.2.106-1 \
    libcusolver-devel-12-1-11.4.5.107-1 \
    libcusparse-devel-12-1-12.1.0.106-1
ln -s cuda-12.1 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
