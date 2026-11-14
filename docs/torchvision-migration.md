# Torchvision Migration

## Argument Mapping

| Torchvision | tvdcn | Notes |
| --- | --- | --- |
| `deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None)` | `deform_conv2d(input, weight, offset=None, mask=None, bias=None, stride=1, padding=0, dilation=1, groups=1)` | Argument order differs; tvdcn exposes convolution groups explicitly. Torchvision infers them from weight. |
| `DeformConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)` | Same arguments in `tvdcn.DeformConv2d` | tvdcn also accepts `padding_mode`, `device`, `dtype`. |
| `module(input, offset, mask=None)` | `module(input, offset, mask=None)` | Pass `None` explicitly to omit offsets in tvdcn. |

Both infer offset groups from offset channels. For torchvision parity, use the same number of mask and offset groups. tvdcn allows these group counts to differ.

```python
import torch
import torchvision
import tvdcn
torch.manual_seed(4)
x = torch.randn(2, 4, 7, 7, dtype=torch.float64)
w = torch.randn(6, 2, 3, 3, dtype=torch.float64)
offset = torch.randn(2, 36, 5, 5, dtype=torch.float64) * 0.1
mask = torch.rand(2, 18, 5, 5, dtype=torch.float64)
bias = torch.randn(6, dtype=torch.float64)
a = torchvision.ops.deform_conv2d(x, offset, w, bias, mask=mask)
b = tvdcn.deform_conv2d(x, w, offset, mask, bias, groups=2)
torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
```
