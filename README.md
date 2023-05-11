Torchvision Deformable Convolution Networks
========
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/tvdcn/build_wheels.yml)
![GitHub](https://img.shields.io/github/license/inspiros/tvdcn)

This package contains the PyTorch implementations of the **2D Deformable Convolution** operation
(the commonly used  `torchvision.ops.deform_conv2d`) proposed in https://arxiv.org/abs/1811.11168,
as well as its **1D** and **3D** equivalences, which are not available in `torchvision` (thus the name).

And beyond that, the package also provides the **transposed** versions of them,
which interestingly noone has ever proposed to use.
The main idea is, while `offset` in deformable convolution guides the convolution kernel where to get the inputs to
compute the output; in transposed deformable convolution, it guides the convolution kernel where to write the outputs.

## Highlights

- **Supported operations:** _(All operations are implemented in C++/Cuda)_

  - `tvdcn.ops.deform_conv1d`
  - `tvdcn.ops.deform_conv2d`
  - `tvdcn.ops.deform_conv3d`
  - `tvdcn.ops.deform_conv_transpose1d`
  - `tvdcn.ops.deform_conv_transpose2d`
  - `tvdcn.ops.deform_conv_transpose3d`

- `offset` and `mask` can be applied in separate groups
  (following [Deformable Convolution v3](https://arxiv.org/abs/2211.05778)).

- All the `nn.Module` wrappers for these operations are implemented,
  everything is `@torch.jit.script`-able! Please check [Usage](#usage).

**Note:** We don't care much about `onnx` exportation, but if you do, you can check this repo:
https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter.

## Requirements

- `torch>=1.8`

## Installation

#### From TestPyPI:

[tvdcn](https://test.pypi.org/project/tvdcn) provides some prebuilt wheels at **TestPyPI**.
Run this command to install:

```terminal
pip install --index-url https://test.pypi.org/simple/ tvdcn
```

The Linux and Windows wheels are built with Cuda 11.8, which is compatible with `torch==2.0.0`.
If you cannot find a wheel for your Arch/Python/Cuda, or there is any problem with library linking when importing,
please proceed to [instructions to build from source](#from-source), all steps are super easy.

|                 |     Linux/Windows     |  MacOS   |
|-----------------|:---------------------:|:--------:|
| Python version: |       3.8-3.11        | 3.8-3.11 |
| Cuda version:   |         11.8          |    -     |
| GPU CCs:        | `6.1,7.5,8.6,8.9+PTX` |    -     |

#### From Source:

For installing from source, you need a C++ compiler (`gcc`/`msvc`) and a Cuda compiler (`nvcc`).
Clone this repo and execute the following command:

```terminal
pip install .
```

Or just compile the binary for inplace usage:

```terminal
python setup.py build_ext --inplace
```

A binary (`.so` file for Unix and `.pyd` file for Windows) should be compiled inside the `tvdcn` folder.
To check if installation is successful, try:

```python
from tvdcn import _HAS_OPS

assert _HAS_OPS
```

**Note:** We use soft Cuda version compatibility checking between the built binary and the installed PyTorch,
which means only major version matching is required.
However, we suggest building the binaries with the same Cuda version with installed PyTorch's Cuda version to prevent
any possible conflict.

## Usage

#### Functions:

Functionally, the package offers 6 functions (listed in [Highlights](#highlights)) much similar to
`torchvision.ops.deform_conv2d`.
However, the order of parameters is slightly different, so be cautious.
Specifically, the signatures of `deform_conv2d` and `deform_conv_transpose2d` look like this:

```python
def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1) -> Tensor:
    ...


def deform_conv_transpose2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        output_padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1) -> Tensor:
    ...
```

If `offset=None` and `mask=None`, the executed operations are identical to conventional convolution.

#### Neural Network Layers:

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

conv = DeformConv2d(3, 16, kernel_size=(3, 3))

output = conv(input, offset, mask)
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
The only tunable hyperparameters that effect these supplementary conv layers are `offset_groups` and `mask_groups`,
which have been decoupled from and behave somewhat similar to `groups`.

```python
import torch

from tvdcn import PackedDeformConv1d

input = torch.rand(2, 3, 128)

conv = PackedDeformConv1d(3, 16, kernel_size=5, modulated=True)
# jit scripting
conv_jit = torch.jit.script(conv)
print(conv_jit)

output = conv_jit(input)
print(output.shape)
```

**Note:** For transposed packed modules, we are generating `offset` and `mask` with pointwise convolution
as we haven't found a better way to do it.

Check the [examples](examples) folder, maybe you can find something helpful.

## Acknowledgements

This for fun project is directly modified and extended from `torchvision.ops.deform_conv2d`.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
