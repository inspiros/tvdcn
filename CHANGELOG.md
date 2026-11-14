# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 2.0.0 (Unreleased)

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

### Removed

- Remove the legacy non-stable-ABI build and native autograd and autocast kernels.
