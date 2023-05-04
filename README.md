Torchvision Deformable Convolution Networks
========

This package contains the PyTorch implementations of the **2D Deformable Convolution** operation
(a commonly used  `torchvision.ops.deform_conv2d`) proposed in https://arxiv.org/abs/1811.11168,
as well as its **1D** and **3D** equivalences, which are not available in `torchvision` (thus the name).

And beyond that, we also provide the **transposed** versions of them, which interestingly noone has ever used.
Is that because they are programmatically challenging 😉?

## Highlights:

**Supported operations:** _(All operations are implemented in C++/CUDA)_
- `deform_conv1d`
- `deform_conv2d`
- `deform_conv3d`
- `deform_conv_transpose1d`
- `deform_conv_transpose2d`
- `deform_conv_transpose3d`

Besides, all the `nn.Module` wrappers for these operations are implemented, please check [Usage](#usage).

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

Functionally, we produced 6 `@torch.jit.script`-able functions (listed in [Highlights](#highlights)) much similar to `torchvision.ops.deform_conv2d`.
However, the order of parameters is slightly different, be cautious:

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

Their raw `nn.Module` wrappers are:
- `DeformConv1d`
- `DeformConv2d`
- `DeformConv3d`
- `DeformConvTranspose1d`
- `DeformConvTranspose2d`
- `DeformConvTranspose3d`

They are subclasses of the `torch.nn.modules._ConvNd` and `torch.nn.modules._ConvTransposeNd`,
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
- `PackedDeformConv1d`
- `PackedDeformConv2d`
- `PackedDeformConv3d`
- `PackedDeformConvTranspose1d`
- `PackedDeformConvTranspose2d`
- `PackedDeformConvTranspose3d`

These are easy-to-use classes that contain the appropriate to generate
`offset` (and `mask` if initialized with `modulated=True`) using ordinary convolution layers;
but note that they offer less customizations.

```python
import torch

from tvdcn import PackedDeformConv1d

input = torch.rand(2, 3, 128)

conv = PackedDeformConv1d(3, 16, kernel_size=5, modulated=True)

output = conv(input)
print(output.shape)
```

Check the [examples](examples) folder, maybe you can find something helpful.

## Acknowledgements

This for fun project is directly modified and extended from `torchvision.ops.deform_conv2d`.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
