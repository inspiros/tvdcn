# Mask Activations

`tvdcn.mask_sigmoid(mask, scale=2.0)` returns `scale * sigmoid(mask)`; `scale` must be positive. `MaskSigmoid(scale=2.0)` is its module form. The default maps zero to one.

`tvdcn.mask_softmax1d(mask, kernel_size)`, `mask_softmax2d`, and `mask_softmax3d` normalize across the kernel positions independently for each batch, mask group, and spatial position. Masks have shape `(N, mask_groups * product(kernel_size), *spatial)` and must be batched; the channel count must be a positive multiple of the kernel volume. Kernel size may be an integer or a tuple of the matching dimensionality. The corresponding `MaskSoftmax1d/2d/3d(kernel_size)` modules accept `forward(mask, kernel_size=None)`, allowing a per-call override.

```python
import torch
from tvdcn import mask_sigmoid, MaskSoftmax2d
mask = torch.zeros(1, 18, 4, 4)
assert torch.all(mask_sigmoid(mask) == 1)
normalized = MaskSoftmax2d(3)(mask).view(1, 2, 9, 4, 4)
torch.testing.assert_close(normalized.sum(dim=2), torch.ones(1, 2, 4, 4))
```
