#!/bin/bash
# Build inside manylinux_2_28 from the checkout root.
set -euxo pipefail
: "${TORCH_VERSION:?}" "${TORCH_VARIANT:?}"
export PATH="/opt/python/${PYTHON_TAG:-cp311-cp311}/bin:$PATH"
if [ "$TORCH_VARIANT" != cpu ]; then
  curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
  dnf install -y cuda-nvcc-13-0 cuda-cccl-13-0 cuda-cudart-devel-13-0 libcublas-devel-13-0 libcurand-devel-13-0 libcusolver-devel-13-0 libcusparse-devel-13-0
  dnf clean all
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  nvcc -V
  source .github/workflows/cuda/Linux-env.sh
else
  export TVDCN_BUILD_CUDA=0
fi
python -m pip install --upgrade setuptools wheel ninja auditwheel
python -m pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python setup.py bdist_wheel --dist-dir=dist
# Strip the packaged modules: bdist_wheel may relink after build_ext.
python -m wheel unpack dist/*.whl -d wheel-unpacked
find wheel-unpacked -name '*.so' -exec ls -lh {} \; -exec strip --strip-debug {} \; -exec ls -lh {} \;
mkdir -p stripped
python -m wheel pack wheel-unpacked/* -d stripped
rm -r wheel-unpacked
# Torch and CUDA dependencies are provided by the installed torch distribution.
excludes=()
while IFS= read -r library; do
  case "$library" in
    libtorch*.so*|libc10*.so*|libcudart.so*|libcublas*.so*|libcufft*.so*|libcurand*.so*|libcusolver*.so*|libcusparse*.so*|libnv*.so*|libcudnn*.so*)
      excludes+=(--exclude "$library") ;;
  esac
done < <(find build/lib* -name '*.so' -exec readelf -d {} \; | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p' | sort -u)
auditwheel repair "${excludes[@]}" --plat manylinux_2_28_x86_64 -w wheelhouse stripped/*.whl
python tools/packaging/check_stable_abi.py wheelhouse/*.whl
rm -r dist stripped
