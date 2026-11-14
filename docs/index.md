# tvdcn Documentation

tvdcn extends `torchvision.ops.deform_conv2d` with 1D and 3D operators, transposed operators, independent offset and mask groups, optional offsets and masks, mask activations, and packed modules that generate offsets and masks. CPU and CUDA kernels support float16, bfloat16, float32 and float64.

| Wheel Platform | CPU | CUDA |
| --- | --- | --- |
| Linux x86_64 | Yes | CUDA 13.0 |
| Windows | Yes | CUDA 13.0 |
| macOS arm64 | Yes | No |

See [Installation](installation.md), [Operators](operators.md), [Modules](modules.md), [Mask Activations](mask-activations.md), [Torchvision Migration](torchvision-migration.md), [Compile And Export](compile-and-export.md), and [C++ API](cpp-api.md). See the [Changelog](../CHANGELOG.md) for release changes.
