# Compile And Export

The operators have Python-side autograd, autocast and fake registrations. Modules can be used with `torch.compile(fullgraph=True)` and `torch.export`:

```python
import torch
from tvdcn import DeformConv2d
layer = DeformConv2d(3, 4, 3)
x = torch.randn(1, 3, 8, 8)
offset = torch.randn(1, 18, 6, 6)
compiled = torch.compile(layer, fullgraph=True)
torch.testing.assert_close(compiled(x, offset), layer(x, offset))
exported = torch.export.export(layer, (x, offset))
torch.testing.assert_close(exported.module()(x, offset), layer(x, offset))
```

`torch.library.opcheck` checks custom operator registration behavior, including fake and autograd registrations:

```python
import torch
import tvdcn
x = torch.randn(1, 2, 5, 5, requires_grad=True)
w = torch.randn(3, 2, 3, 3, requires_grad=True)
result = torch.library.opcheck(torch.ops.tvdcn.deform_conv2d.default, (x, w))
assert all(value == 'SUCCESS' for value in result.values())
```

## TorchScript And ONNX

On CPU with PyTorch 2.14.1, `torch.jit.script` and `torch.jit.trace` both ran successfully for `DeformConv2d` with input, offset and mask, and for default `PackedDeformConv2d` with input. This is an observation for those configurations; tracing records the path exercised by its example inputs. PyTorch emits deprecation warnings for these JIT APIs. ONNX export is not supported.
