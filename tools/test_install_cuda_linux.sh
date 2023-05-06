#! /bin/bash

set -e
set -x

# Install CUDA 11.8, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
#yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
#yum install --setopt=obsoletes=0 -y \
#    cuda-nvcc-11-8-11.8.89-1 \
#    cuda-cudart-devel-11-8-11.8.89-1 \
#    libcurand-devel-11-8-10.3.0.86-1 \
#    libcublas-devel-11-8-11.11.3.6-1
#ln -s cuda-11.8 /usr/local/cuda

add-apt-repository https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
sudo apt-get install -y \
    cuda-nvcc-11-2-11.2.152-1 \
    cuda-cudart-devel-11-2-11.2.152-1 \
    libcurand-devel-11-2-10.2.3.152-1 \
    libcudnn8-devel-8.1.1.33-1.cuda11.2 \
    libcublas-devel-11-2-11.4.1.1043-1
ln -s cuda-11.2 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"
