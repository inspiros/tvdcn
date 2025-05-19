#! /bin/bash

set -e
set -x

# Install CUDA 12.8:
echo "Installing CUDA 12.8"

OS=ubuntu2404

wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get -y install cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 libcurand-dev-12-8 libcusolver-dev-12-8 libcusparse-dev-12-8
sudo apt clean

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
