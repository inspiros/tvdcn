from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='tvdcn-packaging-stable-test',
    version='0.0.0',
    ext_modules=[CppExtension(
        '_stable_ext', ['stable_ext.cpp'], py_limited_api=True,
        define_macros=[('TORCH_TARGET_VERSION', '0x020e000000000000'),
                       ('Py_LIMITED_API', '0x030A0000')],
    )],
    cmdclass={'build_ext': BuildExtension},
    options={'bdist_wheel': {'py_limited_api': 'cp310'}},
)
