Torchvision Deformable Convolution Networks
========

This package contains the PyTorch implementations of the **2D Deformable Convolution** operation
(the commonly used  `torchvision.ops.deform_conv2d`) proposed in https://arxiv.org/abs/1811.11168,
as well as its **1D** and **3D** equivalences, which are not available in `torchvision` (thus the name).

And beyond that, we also provide the **transposed** versions of them,
which interestingly noone has ever proposed to use.
Is that because they are programmatically challenging 😉?

## Highlights:

**Supported operations:** _(All operations are implemented in C++/CUDA)_
- `tvdcn.ops.deform_conv1d`
- `tvdcn.ops.deform_conv2d`
- `tvdcn.ops.deform_conv3d`
- `tvdcn.ops.deform_conv_transpose1d`
- `tvdcn.ops.deform_conv_transpose2d`
- `tvdcn.ops.deform_conv_transpose3d`

Besides, all the `nn.Module` wrappers for these operations are implemented,
everything is `@torch.jit.script`-able! Please check [Usage](#usage).

**Note:** We don't care much about `onnx` exportation, but if you do, you can check this repo:
https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter.

## Requirements:

- `torch>=1.8`

## Installation:

For installing from source, you need a C++14 compiler (`gcc`, `msvc`) and a CUDA compiler (`nvcc`) to be installed.
Clone this repo and execute the following commands in terminal:

```terminal
python setup.py build_ext --inplace
```

A binary (`.so` file for Unix and `.pyd` file for Windows) should be compiled inside `tvdcn` folder.
To check if installation is successful, try:

```python
from tvdcn import _HAS_OPS

assert _HAS_OPS
```

**Note:** We use soft CUDA version compatibility checking between the built binary and the installed PyTorch,
which means only major version matching is required.
However, you better build the binaries with the same CUDA version with installed PyTorch's CUDA version to prevent
any possible incompability.

## Usage:

Functionally, the package offers 6 functions (listed in [Highlights](#highlights)) much similar to
`torchvision.ops.deform_conv2d`.
However, the order of parameters is slightly different, so be cautious:

```python
def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: IntTuple = (1, 1),
        padding: IntTuple = (0, 0),
        dilation: IntTuple = (1, 1),
        groups: int = 1) -> Tensor:
    ...
```

The `nn.Module` wrappers are:
- `tvdcn.ops.DeformConv1d`
- `tvdcn.ops.DeformConv2d`
- `tvdcn.ops.DeformConv3d`
- `tvdcn.ops.DeformConvTranspose1d`
- `tvdcn.ops.DeformConvTranspose2d`
- `tvdcn.ops.DeformConvTranspose3d`

They are subclasses of the `torch.nn.modules._ConvNd`,
but you have to specify `offset` and optionally `mask` as extra inputs for the `forward` function.
For example:

```python
import torch

from tvdcn import DeformConv2d

input = torch.rand(2, 3, 64, 64)
offset = torch.rand(2, 2 * 3 * 3, 62, 62)
# if mask is None, perform the original deform_conv without modulation (v2)
mask = torch.rand(2, 1 * 3 * 3, 62, 62)

layer = DeformConv2d(3, 16, kernel_size=(3, 3))

output = layer(input, offset, mask)
print(output.shape)
```

Additionally, following many other implementations out there, we also implemented the _packed_ wrappers:
- `tvdcn.ops.PackedDeformConv1d`
- `tvdcn.ops.PackedDeformConv2d`
- `tvdcn.ops.PackedDeformConv3d`
- `tvdcn.ops.PackedDeformConvTranspose1d`
- `tvdcn.ops.PackedDeformConvTranspose2d`
- `tvdcn.ops.PackedDeformConvTranspose3d`

These are easy-to-use classes that contain ordinary convolution layers with appropriate hyperparameters to generate
`offset` (and `mask` if initialized with `modulated=True`);
but that means less customization.

```python
import torch

from tvdcn import PackedDeformConv1d

input = torch.rand(2, 3, 128)

conv = PackedDeformConv1d(3, 16, kernel_size=5, modulated=True)

output = conv(input)
print(output.shape)
```

**Note:** For transposed packed modules, we are generating `offset` and `mask` with pointwise convolution
as we haven't found a better way to do it.

Check the [examples](examples) folder, maybe you can find something helpful.

## Acknowledgements

This for fun project is directly modified and extended from `torchvision.ops.deform_conv2d`.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
