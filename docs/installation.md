# Installation

## Requirements

Python 3.10 or later and PyTorch 2.14 or later are required. Linux x86_64 and Windows wheels include CUDA 13.0 kernels; macOS arm64 wheels are CPU only. CUDA use requires an NVIDIA driver supporting CUDA 13 and a CUDA-enabled PyTorch installation. Wheels target `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` and PTX.

```console
pip install tvdcn
```

## Verify Installation

```python
import tvdcn
print('Native operators:', tvdcn.has_ops())
print('CUDA operators:', tvdcn.with_cuda())
print('Compiled CUDA version:', tvdcn.cuda_version())
print('CUDA architecture flags:', tvdcn.cuda_arch_list())
```

`cuda_version()` returns an integer such as `13000`, or `-1` when CUDA operators are unavailable; `cuda_arch_list()` returns an empty list in that case. `has_ops()` checks the core native extension, independently of CUDA.

## Build From Source

Install a C++ compiler and PyTorch in the build environment. For CUDA, install a CUDA toolkit with `nvcc`, set `CUDA_HOME` if necessary, and use a CUDA-enabled PyTorch. Build with:

```console
pip install . --no-build-isolation
```

For an in-place build use `python setup.py build_ext --inplace`. For a CUDA 12.x PyTorch build, such as cu126, install that PyTorch build first and build tvdcn from source with a compatible CUDA 12.x toolkit. The extension and PyTorch must have the same CUDA major version. This is also the path for drivers that cannot run CUDA 13 wheels.

## Build Options

| Environment Variable | Behavior |
| --- | --- |
| `TVDCN_BUILD_CUDA` | `auto` (default) builds CUDA when `CUDA_HOME` and CUDA-enabled torch are present, without requiring a visible GPU; `0` disables it; `1` requires both. |
| `TVDCN_STABLE_ABI` | Unset or `1` targets the PyTorch 2.14 stable ABI and CPython 3.10 limited API; `0` is rejected, as are other values. |
| `TVDCN_LOCAL_VERSION` | A nonempty label appends `+<label>` to the PEP 621 version. |
| `TVDCN_CUDA_ARCH_LIST` | Overrides `TORCH_CUDA_ARCH_LIST`; otherwise PyTorch chooses the architectures. |

`FORCE_CUDA=1` is a deprecated alias for `TVDCN_BUILD_CUDA=1` when the latter is unset. `DEBUG=1` enables debug compilation; `NVCC_FLAGS` supplies additional nvcc flags. See [setup.py](../setup.py) for implementation details.
