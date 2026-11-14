# Build options (environment variables):
# TVDCN_BUILD_CUDA=auto|0|1: auto uses CUDA_HOME and a CUDA-enabled torch,
#   without requiring an accessible GPU. FORCE_CUDA=1 is a deprecated alias.
# TVDCN_STABLE_ABI: unset or 1 targets torch 2.14 and CPython 3.10's limited
#   API (cp310-abi3 wheels). 0 is rejected: the legacy build was removed in 2.0.
# TVDCN_LOCAL_VERSION=<label>: append +<label> to the PEP 621 version.
# TVDCN_CUDA_ARCH_LIST: override TORCH_CUDA_ARCH_LIST; otherwise torch chooses.
# DEBUG=1 and NVCC_FLAGS retain their usual debug/compiler flag behavior.
import glob
import importlib.machinery
import os
import shlex
import sys
import warnings
from pathlib import Path

import torch
from setuptools import Distribution, setup
from packaging.version import Version
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension

PACKAGE_ROOT = 'tvdcn'


class TVDCNDistribution(Distribution):
    def parse_config_files(self, *args, **kwargs):
        super().parse_config_files(*args, **kwargs)
        label = os.getenv('TVDCN_LOCAL_VERSION')
        if label:
            self.metadata.version = str(Version(f'{self.metadata.version}+{label}'))


class TVDCNBuildExtension(torch.utils.cpp_extension.BuildExtension):
    def run(self):
        super().run()
        # Switching CUDA or ABI modes must not package/load an old in-place
        # binary, or an old binary left in build/lib by a previous wheel.
        expected = {ext.name.rsplit('.', 1)[1]: self.get_ext_filename(ext.name).split(os.sep)[-1]
                    for ext in self.extensions}
        directories = [Path(self.build_lib) / PACKAGE_ROOT]
        if self.inplace:
            directories.append(Path(PACKAGE_ROOT))
        for directory in directories:
            for name in ('_C', '_C_cuda'):
                for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                    path = directory / (name + suffix)
                    if path.name != expected.get(name) and path.exists():
                        path.unlink()


def get_parallel_options(backend=None):
    parallel_extra_compile_args = []
    parallel_define_macros = []

    if backend is not None:
        if backend.upper() not in ['OPENMP', 'NATIVE', 'NATIVE_TBB']:
            raise ValueError('Parallel backend options are OPENMP, NATIVE, or NATIVE_TBB. '
                             f'Got unknown backend {backend}.')
    else:  # detect torch parallel backend
        parallel_info_string = torch.__config__.parallel_info()
        parallel_info_array = parallel_info_string.splitlines()
        backend_lines = [line for line in parallel_info_array
                         if line.startswith('ATen parallel backend:')]
        if len(backend_lines):
            backend = backend_lines[0].rsplit(': ')[1]

    backend = backend.lower() if backend is not None else ''
    if backend == 'openmp':
        parallel_define_macros += [('AT_PARALLEL_OPENMP', None)]
        if sys.platform == 'darwin':
            parallel_extra_compile_args.append('-Xpreprocessor')
        parallel_extra_compile_args.append('/openmp' if sys.platform == 'win32' else '-fopenmp')
        if sys.platform == 'darwin':
            parallel_extra_compile_args.append('-lomp')
    elif backend.startswith('native'):
        if backend.endswith('tbb'):
            parallel_define_macros += [('AT_PARALLEL_NATIVE_TBB', None)]
        else:
            parallel_define_macros += [('AT_PARALLEL_NATIVE', None)]
    return parallel_extra_compile_args, parallel_define_macros


def get_extensions():
    extensions_dir = os.path.join(PACKAGE_ROOT, 'csrc')
    sources = [os.path.join(extensions_dir, 'tvdcn.cpp')]
    for directory in ('ops', 'ops/cpu'):
        sources += sorted(glob.glob(os.path.join(extensions_dir, directory, '*.cpp')))
    cuda_sources = [os.path.join(extensions_dir, 'tvdcn_cuda.cpp')]
    for suffix in ('*.cu', '*.cpp'):
        cuda_sources += sorted(glob.glob(os.path.join(extensions_dir, 'ops/cuda', suffix)))

    cuda_mode = os.getenv('TVDCN_BUILD_CUDA', 'auto')
    if cuda_mode not in ('auto', '0', '1'):
        raise ValueError('TVDCN_BUILD_CUDA must be auto, 0, or 1')
    if os.getenv('FORCE_CUDA') == '1':
        warnings.warn('FORCE_CUDA=1 is deprecated; use TVDCN_BUILD_CUDA=1',
                      FutureWarning)
        if 'TVDCN_BUILD_CUDA' not in os.environ:
            cuda_mode = '1'
    build_cuda = cuda_mode == '1' or (
        cuda_mode == 'auto' and CUDA_HOME is not None and torch.version.cuda is not None)
    if build_cuda and (CUDA_HOME is None or torch.version.cuda is None):
        raise RuntimeError('CUDA builds require CUDA_HOME and a CUDA-enabled torch')
    stable_mode = os.getenv('TVDCN_STABLE_ABI', '1')
    if stable_mode == '0':
        raise RuntimeError('TVDCN_STABLE_ABI=0 is unsupported: the legacy build was removed in 2.0. '
                           'Unset TVDCN_STABLE_ABI or set it to 1 to build the stable ABI.')
    if stable_mode != '1':
        raise ValueError('TVDCN_STABLE_ABI must be unset or 1')
    if 'TVDCN_CUDA_ARCH_LIST' in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ['TVDCN_CUDA_ARCH_LIST']

    debug = os.getenv('DEBUG', '0') == '1'
    print(f'Build options: CUDA={build_cuda} ({cuda_mode}), stable ABI=True, DEBUG={debug}')
    cxx_flags, macros = get_parallel_options('openmp')
    # torch 2.14's BuildExtension selects C++20 for both cxx and nvcc.
    nvcc_flags = shlex.split(os.getenv('NVCC_FLAGS', ''))
    if build_cuda:
        macros += [('USE_CUDA', None)]
    if sys.platform != 'win32':
        cxx_flags += ['-fvisibility=hidden']
        nvcc_flags += ['-Xcompiler=-fvisibility=hidden']
    macros += [('TORCH_TARGET_VERSION', '0x020e000000000000'),
               ('Py_LIMITED_API', '0x030A0000')]
    if sys.platform == 'win32':
        macros += [(f'{PACKAGE_ROOT}_EXPORTS', None), ('USE_PYTHON', None)]
        cxx_flags.append('/MP')
    if debug:
        cxx_flags += ['/Od', '/Zi'] if sys.platform == 'win32' else ['-g', '-O0']
        nvcc_flags = [f for f in nvcc_flags if not f.startswith(('-O', '-g'))]
        nvcc_flags += ['-O0', '-g']

    def make_extension(factory, name, extension_sources, cuda=False):
        flags = {'cxx': list(cxx_flags)}
        extension_macros = list(macros)
        if cuda:
            flags['nvcc'] = list(nvcc_flags)
            # Record the actual flags torch will use (including auto detection).
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            archs = ' '.join(_get_cuda_arch_flags(nvcc_flags))
            extension_macros += [('TVDCN_CUDA_ARCH_FLAGS', f'"{archs}"')]
        return factory(
            f'{PACKAGE_ROOT}.{name}', sources=extension_sources,
            include_dirs=[extensions_dir], define_macros=extension_macros,
            extra_compile_args=flags, py_limited_api=True)

    extensions = [make_extension(CppExtension, '_C', sources)]
    if build_cuda:
        extensions.append(make_extension(CUDAExtension, '_C_cuda', cuda_sources, cuda=True))
    return extensions


def setup_package():
    setup(
        distclass=TVDCNDistribution,
        ext_modules=get_extensions(),
        options={'bdist_wheel': {'py_limited_api': 'cp310'}},
        cmdclass={
            'build_ext': TVDCNBuildExtension
        },
    )


if __name__ == '__main__':
    setup_package()
