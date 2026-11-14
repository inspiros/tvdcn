Torchvision+ Deformable Convolution Networks
========
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/tvdcn/build_wheels.yml)](https://github.com/inspiros/tvdcn/actions)
[![PyPI](https://img.shields.io/pypi/v/tvdcn)](https://pypi.org/project/tvdcn)
[![Downloads](https://static.pepy.tech/badge/tvdcn)](https://pepy.tech/project/tvdcn)
[![GitHub](https://img.shields.io/github/license/inspiros/tvdcn)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/355479452.svg)](https://doi.org/10.5281/zenodo.14699341)

This package contains PyTorch implementations of **Deformable Convolution** (the commonly used
`torchvision.ops.deform_conv2d`) proposed in https://arxiv.org/abs/1811.11168, and **Transposed Deformable Convolution**
proposed in https://arxiv.org/abs/2210.09446 (currently without interpolation kernel scaling). It also supports their
**1D** and **3D** equivalents, which are not available in `torchvision` (thus the name).

## Highlights

- **Supported operators:** _(implemented in C++/CUDA)_
    - `tvdcn.ops.deform_conv1d`, `tvdcn.ops.deform_conv2d`, `tvdcn.ops.deform_conv3d`
    - `tvdcn.ops.deform_conv_transpose1d`, `tvdcn.ops.deform_conv_transpose2d`,
      `tvdcn.ops.deform_conv_transpose3d`
- **Supplementary operators:** `tvdcn.ops.mask_softmax1d`, `mask_softmax2d`, and `mask_softmax3d` (the mask activation
  proposed in https://arxiv.org/abs/2211.05778).
- Offsets, masks, and bias are optional. Offset and mask groups can be configured separately from convolution groups.
- Packed modules can generate offsets and masks, with sigmoid or softmax mask activations.
- Supports float32, float64, float16, and bfloat16 on CPU and CUDA.
- Autograd, autocast, and fake kernels are registered with `torch.library`. Operators work with
  `torch.compile(fullgraph=True)` and `torch.export`.

**Note:** `tvdcn` doesn't support ONNX export.

## What's New in 2.0

Version 2.0 uses the PyTorch stable ABI. Operator names and schemas are unchanged from 1.x. PyPI provides one `abi3`
wheel per platform, compatible with Python 3.10 and later and with PyTorch 2.14 and later. The C++ API is still available through the header-inline wrappers in [`tvdcn/csrc/ops`](tvdcn/csrc/ops);
see the [C++ API smoke example](tests/packaging/cpp_api/smoke.cpp).

## Requirements

- Python >= 3.10
- PyTorch >= 2.14

## Installation

### From PyPI

Install tvdcn with pip:

```console
pip install tvdcn
```

PyPI wheels are available for Linux x86_64 and Windows with CPU and CUDA kernels, and for macOS arm64 with CPU kernels.
The Linux and Windows CUDA kernels are built with CUDA 13.0 for `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, and
`sm_120`, plus PTX. CUDA kernels load only when PyTorch has CUDA; the same wheel works on CPU with a CPU-only PyTorch
installation. For CUDA 12 or older drivers, build tvdcn from source.

### From Source

Install a C++ compiler and, for CUDA kernels, a CUDA toolkit with `nvcc`. With PyTorch installed in the build
environment, run:

```console
pip install . --no-build-isolation
```

For in-place use, run:

```console
python setup.py build_ext --inplace
```

Build options are documented at the top of [`setup.py`](setup.py), including `TVDCN_BUILD_CUDA`,
`TVDCN_CUDA_ARCH_LIST`, and `TVDCN_LOCAL_VERSION`.

## Usage

#### Operators

The package offers six deformable convolution functions, listed in [Highlights](#highlights). Their argument order
differs slightly from `torchvision.ops.deform_conv2d`; see [the migration guide](docs/torchvision-migration.md).

```python
import torch
from tvdcn.ops import deform_conv2d

x = torch.rand(1, 3, 8, 8)
weight = torch.rand(4, 3, 3, 3)
offset = torch.rand(1, 18, 6, 6)
mask = torch.rand(1, 9, 6, 6).sigmoid()
y = deform_conv2d(x, weight, offset, mask)
print(y.shape)
```

If `offset=None` and `mask=None`, the operation is identical to conventional convolution.

The transposed operators take offsets and masks with the spatial size of the input:

```python
import torch
from tvdcn.ops import deform_conv_transpose2d

x = torch.rand(1, 4, 6, 6)
weight = torch.rand(4, 3, 3, 3)  # (in_channels, out_channels // groups, kH, kW)
offset = torch.rand(1, 18, 6, 6)
mask = torch.rand(1, 9, 6, 6).sigmoid()
y = deform_conv_transpose2d(x, weight, offset, mask, stride=2)
print(y.shape)  # torch.Size([1, 3, 13, 13])
```

#### Neural Network Layers

The `nn.Module` wrappers are `DeformConv1d/2d/3d` and `DeformConvTranspose1d/2d/3d`, available from `tvdcn`. Pass
offsets and optionally masks to their `forward` method:

```python
import torch
from tvdcn import DeformConv2d

conv = DeformConv2d(3, 4, kernel_size=3)
x = torch.rand(1, 3, 8, 8)
offset = torch.rand(1, 18, 6, 6)
mask = torch.rand(1, 9, 6, 6).sigmoid()

y = conv(x, offset, mask)
print(y.shape)
```

The `PackedDeformConv1d/2d/3d` and `PackedDeformConvTranspose1d/2d/3d` modules generate offsets (and optionally masks)
with convolution layers. Configure `offset_groups`, `mask_groups`, `deformable`, and `modulated` as needed. For example,
`mask_activation='softmax'` selects softmax mask activation; `offset_activation` and `mask_activation` also accept
`nn.Module` instances.

Modules can be compiled or exported with PyTorch:

```python
import torch
from tvdcn import DeformConv2d

conv = DeformConv2d(3, 4, kernel_size=3)
x = torch.rand(1, 3, 8, 8)
offset = torch.rand(1, 18, 6, 6)
mask = torch.rand(1, 9, 6, 6).sigmoid()

compiled = torch.compile(conv, fullgraph=True)
print(compiled(x, offset, mask).shape)

exported = torch.export.export(conv, (x, offset, mask))
print(exported.module()(x, offset, mask).shape)
```

For transposed packed modules, offsets and masks are generated with pointwise convolution.

See the [examples](examples) folder for more.

## Documentation

See the [documentation](docs/index.md) and [changelog](CHANGELOG.md).

## Acknowledgements

This _for fun_ project is directly modified and extended from `torchvision.ops.deform_conv2d`.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
