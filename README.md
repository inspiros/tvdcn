Torchvision+ Deformable Convolution Networks
========
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/tvdcn/build_wheels.yml)](https://github.com/inspiros/tvdcn/actions)
[![PyPI](https://img.shields.io/pypi/v/tvdcn)](https://pypi.org/project/tvdcn)
[![Downloads](https://static.pepy.tech/badge/tvdcn)](https://pepy.tech/project/tvdcn)
[![GitHub](https://img.shields.io/github/license/inspiros/tvdcn)](LICENSE.txt)

This package contains the PyTorch implementations of the **Deformable Convolution** operation
(the commonly used  `torchvision.ops.deform_conv2d`) proposed in https://arxiv.org/abs/1811.11168,
and the **Transposed Deformable Convolution** proposed in https://arxiv.org/abs/2210.09446
(currently without interpolation kernel scaling).
It also supports their **1D** and **3D** equivalences, which are not available in `torchvision` (thus the name).

## Highlights

- **Supported operators:** _(All are implemented in C++/Cuda)_
    - `tvdcn.ops.deform_conv1d`
    - `tvdcn.ops.deform_conv2d` _(faster than `torchvision.ops.deform_conv2d` by at least 10% during forward pass
       on our **Quadro RTX 5000** according to [this test](tests/test_speed.py))_
    - `tvdcn.ops.deform_conv3d`
    - `tvdcn.ops.deform_conv_transpose1d`
    - `tvdcn.ops.deform_conv_transpose2d`
    - `tvdcn.ops.deform_conv_transpose3d`

- And the following **supplementary operators** (`mask` activation proposed in https://arxiv.org/abs/2211.05778):
    - `tvdcn.ops.mask_softmax1d`
    - `tvdcn.ops.mask_softmax2d`
    - `tvdcn.ops.mask_softmax3d`

- Both `offset` and `mask` can be turned off, and can be applied in separate groups.

- All the `nn.Module` wrappers for these operators are implemented,
  everything is `@torch.jit.script`-able! Please check [Usage](#usage).

**Note:** We don't care much about `onnx` exportation, but if you do, you can check this repo:
https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter.

## Requirements

- `torch>=2.1.0` (``torch>=1.9.0`` if installed from source)

## Installation

#### From PyPI:

[tvdcn](https://pypi.org/project/tvdcn) provides some prebuilt wheels on **PyPI**.
Run this command to install:

```terminal
pip install tvdcn
```

Since **PyTorch** is migrating to Cuda 12 versions,
our Linux and Windows wheels are built with **Cuda 12.1** and won't be compatible with older versions.

|                  |              Linux/Windows               |     MacOS      |
|------------------|:----------------------------------------:|:--------------:|
| Python version:  |                 3.8-3.11                 |    3.8-3.11    |
| PyTorch version: |              `torch==2.1.0`              | `torch==2.1.0` |
| Cuda version:    |                   12.1                   |       -        |
| GPU CCs:         |  `5.0,6.0,6.1,7.0,7.5,8.0,8.6,9.0+PTX`   |       -        |

When the Cuda versions of ``torch`` and ``tvdcn`` mismatch, you will see an error like this:

```terminal
RuntimeError: Detected that PyTorch and Extension were compiled with different CUDA versions.
PyTorch has CUDA Version=11.8 and Extension has CUDA Version=12.1.
Please reinstall the Extension that matches your PyTorch install.
```

If you see this error instead, that means there are other issues related to Python, PyTorch, device arch, e.t.c.
Please proceed to [instructions to build from source](#from-source), all steps are super easy.

```terminal
RuntimeError: Couldn't load custom C++ ops. Recompile C++ extension with:
     python setup.py build_ext --inplace
```

#### From Source:

For installing from source, you need a C++ compiler (`gcc`/`msvc`) and a Cuda compiler (`nvcc`) with C++17 features
enabled.
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
import tvdcn

print('Library loaded successfully:', tvdcn.has_ops())
print('Compiled with Cuda:', tvdcn.with_cuda())
```

**Note:** We use soft Cuda version compatibility checking between the built binary and the installed PyTorch,
which means only major version matching is required.
However, we suggest building the binaries with the same Cuda version with installed PyTorch's Cuda version to prevent
any possible conflict.

## Usage

#### Operators:

Functionally, the package offers 6 functions (listed in [Highlights](#highlights)) much similar to
`torchvision.ops.deform_conv2d`.
However, the order of parameters is slightly different, so be cautious
(check [this comparison](tests/test_compatibility_with_torchvision.py)).


<table>
<tr>
<th>torchvision</th>
<th>tvdcn</th>
</tr>

<tr>
<td>
<sub>

```python
import torch, torchvision

input = torch.rand(4, 3, 10, 10)
kh, kw = 3, 3
weight = torch.rand(5, 3, kh, kw)
offset = torch.rand(4, 2 * kh * kw, 8, 8)
mask = torch.rand(4, kh * kw, 8, 8)
bias = torch.rand(5)

output = torchvision.ops.deform_conv2d(input, offset, weight, bias,
                                       stride=(1, 1),
                                       padding=(0, 0),
                                       dilation=(1, 1),
                                       mask=mask)
print(output)
```

</sub>
<td>
<sub>

```python
import torch, tvdcn

input = torch.rand(4, 3, 10, 10)
kh, kw = 3, 3
weight = torch.rand(5, 3, kh, kw)
offset = torch.rand(4, 2 * kh * kw, 8, 8)
mask = torch.rand(4, kh * kw, 8, 8)
bias = torch.rand(5)

output = tvdcn.ops.deform_conv2d(input, weight, offset, mask, bias,
                                 stride=(1, 1),
                                 padding=(0, 0),
                                 dilation=(1, 1),
                                 groups=1)
print(output)
```

</sub>
</td>
</tr>

</table>

Specifically, the signatures of `deform_conv2d` and `deform_conv_transpose2d` look like these:

```python
def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1) -> Tensor:
    ...


def deform_conv_transpose2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1) -> Tensor:
    ...
```

If `offset=None` and `mask=None`, the executed operators are identical to conventional convolution.

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

To use the softmax activation for mask proposed in [Deformable Convolution v3](https://arxiv.org/abs/2211.05778),
set `mask_activation='softmax'`. `offset_activation` and `mask_activation` also accept any `nn.Module`.

```python
import torch

from tvdcn import PackedDeformConv1d

input = torch.rand(2, 3, 128)

conv = PackedDeformConv1d(3, 16,
                          kernel_size=5,
                          modulated=True,
                          mask_activation='softmax')
# jit scripting
scripted_conv = torch.jit.script(conv)
print(scripted_conv)

output = scripted_conv(input)
print(output.shape)
```

**Note:** For transposed packed modules, we are generating `offset` and `mask` with pointwise convolution
as we haven't found a better way to do it.

Do check the [examples](examples) folder, maybe you can find something helpful.

## Acknowledgements

This _for fun_ project is directly modified and extended from `torchvision.ops.deform_conv2d`.

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
