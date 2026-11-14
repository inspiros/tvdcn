# Modules

Each family has `1d`, `2d`, and `3d` variants, exported from `tvdcn` and `tvdcn.ops`.

## Constructors

`DeformConvNd(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)` follows the corresponding PyTorch convolution constructor. Replace `Nd` with `1d`, `2d`, or `3d`. `DeformConvTransposeNd` adds `output_padding=0` after `dilation` and before `groups`.

The packed constructors add `offset_groups=1, mask_groups=1` after `groups`, and `generator_bias=False, deformable=True, modulated=False, offset_activation=None, mask_activation='sigmoid'` after `bias`. All other arguments have the same order and defaults as the corresponding unpacked constructor. Both input and output channel counts must be divisible by each packed generator group count. Transposed modules support only zero padding mode.

## Forward Calls

| Family | Forward Signature |
| --- | --- |
| `DeformConvNd` | `forward(input, offset, mask=None)` |
| `DeformConvTransposeNd` | `forward(input, offset, mask=None, output_size=None)` |
| `PackedDeformConvNd` | `forward(input)` |
| `PackedDeformConvTransposeNd` | `forward(input, output_size=None)` |

The unpacked `offset` argument must be passed, but can be `None`. See [operator shapes](operators.md). `output_size` selects a valid transposed output size.

## Packed Generators

`deformable` enables the offset generator; `modulated` enables the mask generator. Disabled generators supply `None`. Regular packed modules use grouped convolutions with the main kernel size, stride, padding and dilation; transposed packed modules use grouped pointwise convolutions. Generator weights and optional biases start at zero. `generator_bias` controls their bias independently of the main convolution bias.

`offset_activation` accepts a callable/module or `None`; strings are rejected. `mask_activation` accepts a callable/module, `None`, `'sigmoid'` (scaled by 2), or `'softmax'`. With zero generator output, scaled sigmoid produces unit masks; softmax produces `1/K` per kernel position. See [Mask Activations](mask-activations.md).

```python
import torch
from tvdcn import PackedDeformConv2d
layer = PackedDeformConv2d(3, 4, 3, modulated=True)
assert layer(torch.randn(2, 3, 8, 8)).shape == (2, 4, 6, 6)
```
