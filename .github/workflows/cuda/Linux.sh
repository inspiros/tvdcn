#! /bin/bash

set -e
set -x

# Install CUDA 12.8:
echo "Installing CUDA 12.8"

OS=ubuntu2404

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-${OS}-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-${OS}-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-${OS}-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt-get -y install cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 libcurand-dev-12-8 libcusolver-dev-12-8 libcusparse-dev-12-8
sudo apt clean

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
