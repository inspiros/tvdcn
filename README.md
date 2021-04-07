# Torchvision Deformable Convolution Networks

*Modified from `torchvision`*

## Check list:

- [x] Runs on both cpu and cuda
- [x] Supports 1D/2D/3D
- [x] Modulated deformable convolution (a.k.a. deformable convolution v2)
- [x] unittest
- [ ] onnx export (no)

## Installation:

### Clone the repo:

```terminal
git clone https://github.com/inspiros/tvdcn.git
```

### Build the required C++ extension

This step requires C++14 compiler (`gcc`, `msvc`) and CUDA compiler (`nvcc`) to be installed. Execute `build_ext.sh` or
run the following commands in terminal:

```terminal
cd tvdcn
python setup.py build_ext --inplace
```

A binary (`.so` file for Unix and `.pyd` file for Windows) should be compiled inside `tvdcn/tvdcn` folder. To check if
installation is successful, try:

```python
from tvdcn._extensions import _HAS_OPS

assert _HAS_OPS
```

### Install

Optionally, install it as a python package:

```terminal
pip install .
```

## Usage:

Functionally, we produce 3 `@torch.jit.script`able functions `deform_conv1d`, `deform_conv2d` and `deform_conv3d` much
similar to `torchvision.ops.deform_conv2d`. However, the order of parameters is slightly different:

```python
def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Tensor,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
) -> Tensor:
    ...
```

The `PackedDeformConv1d`, `PackedDeformConv2d` and `PackedDeformConv3d` are easy-to-use `torch.nn.Module` wrapper
classes that contain the appropriate Conv layers that generates `offset` (and `mask` if initialized with
parameter `modulated=True`) but note that they offer less customizations.

```python
import torch
from tvdcn import PackedDeformConv2d

input = torch.rand(2, 3, 64, 64)

layer = PackedDeformConv2d(3, 16, kernel_size=(3, 3), modulated=True)

output = layer(input)
print(output.shape)
```

Otherwise, the raw `DeformConv1d`, `DeformConv2d` and `DeformConv3d` are available, but you have to specify `offset` and
optionally `mask` as extra inputs.

```python
import torch
from tvdcn import DeformConv2d

input = torch.rand(2, 3, 64, 64)
offset = torch.rand(2, 2 * 3 * 3, 62, 62)
mask = torch.rand(2, 1 * 3 * 3, 62, 62)

layer = DeformConv2d(3, 16, kernel_size=(3, 3))

output = layer(input, offset, mask)
print(output.shape)
```

## Note:

- The latest MSVC update 1928 caused incompatible `floor` function for cuda device code, hence, it is temporarily realiased
  to `floorf` in cuda kernels.
  