# Functional Operators

All six functions are available from `tvdcn` and `tvdcn.ops`. Their signatures (including type annotations) are:

```text
def deform_conv1d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1) -> Tensor:

def deform_conv2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1) -> Tensor:

def deform_conv3d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1) -> Tensor:

def deform_conv_transpose1d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1) -> Tensor:

def deform_conv_transpose2d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1) -> Tensor:

def deform_conv_transpose3d(
        input: Tensor,
        weight: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1) -> Tensor:
```

## Shapes And Groups

For dimension count `D`, let `N` be batch size, `C_in` and `C_out` channel counts, `K = product(kernel_size)`, `G = groups`, `O = offset_groups`, and `M = mask_groups`. Spatial tuples are written `I` for input and `S` for output. Batched shapes are shown; for an unbatched input omit `N` from input, offset, mask and output.

| Tensor | Regular | Transposed |
| --- | --- | --- |
| input | `(N, C_in, *I)` | `(N, C_in, *I)` |
| weight | `(C_out, C_in / G, *kernel_size)` | `(C_in, C_out / G, *kernel_size)` |
| offset | `(N, D * O * K, *S)` | `(N, D * O * K, *I)` |
| mask | `(N, M * K, *S)` | `(N, M * K, *I)` |
| bias | `(C_out,)` | `(C_out,)` |
| output | `(N, C_out, *S)` | `(N, C_out, *S)` |

`G` divides both channel counts. Offset and mask group counts are inferred independently from their channel dimensions, and each must divide `C_in`. They need not equal `G` or each other. For each spatial axis, regular output size is `floor((I + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)`; transposed output size is `(I-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1`.

`offset=None` uses zero displacement; `mask=None` uses unit modulation. Either may be supplied independently. With both absent the operation is ordinary convolution or transposed convolution. Offsets and masks are not activated automatically.

## Offset Layout

Channels are ordered by offset group, then flattened kernel position in spatial order (last axis varies fastest), then coordinate component. The components are width in 1D, `(height, width)` in 2D, and `(depth, height, width)` in 3D; 2D matches torchvision. Mask channels are ordered by mask group then flattened kernel position.

```python
import torch
from tvdcn import deform_conv2d, deform_conv_transpose2d
x = torch.randn(2, 4, 5, 5)
w = torch.randn(6, 2, 3, 3)
o = torch.zeros(2, 2 * 2 * 9, 3, 3)
m = torch.ones(2, 4 * 9, 3, 3)
y = deform_conv2d(x, w, o, m, groups=2)
assert y.shape == (2, 6, 3, 3)
wt = torch.randn(4, 3, 3, 3)
yt = deform_conv_transpose2d(x, wt, torch.zeros(2, 36, 5, 5),
                             torch.ones(2, 36, 5, 5), groups=2)
assert yt.shape == (2, 6, 7, 7)
```
