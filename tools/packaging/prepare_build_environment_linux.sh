#! /bin/bash

set -e
set -x

# Install lapack, blas
yum install -y lapack-devel blas-devel

# Install CUDA 12.4:
echo "Installing CUDA 12.4"
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-12-4-12.4.131-1 \
    cuda-cudart-devel-12-4-12.4.127-1 \
    libcublas-devel-12-4-12.4.5.8-1 \
    libcurand-devel-12-4-10.3.5.147-1 \
    libcusolver-devel-12-4-11.6.1.9-1 \
    libcusparse-devel-12-4-12.3.1.170-1
ln -s cuda-12.4 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
