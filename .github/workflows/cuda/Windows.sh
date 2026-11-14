#!/bin/bash
set -euxo pipefail
# CUDA 13.0 matches the torch cu130 wheels.
CUDA_ROOT='/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0'
curl -fL -o cuda.exe https://developer.download.nvidia.com/compute/cuda/13.0.2/network_installers/cuda_13.0.2_windows_network.exe
status=0
./cuda.exe -s nvcc_13.0 crt_13.0 nvvm_13.0 cudart_13.0 thrust_13.0 cublas_dev_13.0 curand_dev_13.0 cusolver_dev_13.0 cusparse_dev_13.0 -loglevel:6 -log:"$PWD/cuda_install_logs" || status=$?
rm cuda.exe
if [ ! -x "$CUDA_ROOT/bin/nvcc.exe" ]; then
  echo "CUDA install failed (installer exit code $status)"
  cat cuda_install_logs/LOG.setup.exe.log cuda_install_logs/LOG.RunDll32.exe.log 2>/dev/null || ls -R cuda_install_logs || true
  exit 1
fi
rm -rf cuda_install_logs
"$CUDA_ROOT/bin/nvcc.exe" -V
