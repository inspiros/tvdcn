# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 2.0.0 (2026-11-14)

### Breaking

- Require Python 3.10 or later and PyTorch 2.14 or later. CUDA wheels use CUDA 13.0; builds for CUDA 12 require building from source.
- The C++ API now uses stable ABI tensor types and header-inline dispatcher wrappers.

### Added

- bfloat16 CPU and CUDA operator support and Python fake registrations for compile and export.
- CPython 3.10 `abi3` wheels, a stable ABI checker, golden fixtures, packaging tests, and a 2D benchmark.
- CI workflows for wheel builds, tests, and manual publishing.

### Changed

- Port all regular and transposed operators to the PyTorch stable ABI; operator names and schemas remain unchanged.
- Split native libraries into `_C` for CPU and schemas and `_C_cuda` for CUDA. Register autograd and autocast in Python.
- Update packaging and build options and use CUDA 13.0 for release wheels.
- Faster CUDA backward: coalesced col2im/grad kernel indexing, paired float16/bfloat16 atomics and 32-bit indices
  below 2^30 elements. 2D is on par with torchvision in float32; float16 backward is several times faster.
- CPU backward scatter (`col2im` and its 1D/3D counterparts) runs in parallel over (batch, channel) planes.

### Fixed

- Offset and mask gradients for float16/bfloat16 inputs are accumulated in float32 on CPU and CUDA.
- CUDA kernels always used 64-bit indexing because the 32-bit threshold check overflowed.

### Removed

- Remove the legacy non-stable-ABI build and native autograd and autocast kernels.
