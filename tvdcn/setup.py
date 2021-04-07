import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')

        # nvcc_flags.append("-DCUDA_HOST_COMPILER=/usr/bin/gcc-7")
        extra_compile_args["nvcc"] = nvcc_flags

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, os.path.join(extensions_dir, "cpu"), os.path.join(extensions_dir, "cuda")]
    ext_modules = [
        extension(
            "_C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="tvdcn",
    version="1.0",
    author="inspiros",
    author_email='hnhat.tran@gmail.com',
    description="torchvision deformable convolutional networks",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
)
