import torch

from tvdcn import *


def test_wrapper():
    conv = DeformConv2d(3, 4, kernel_size=(2, 2))
    conv = torch.jit.script(conv)
    print(conv)

from setuptools import build_meta
def test_packed_wrapper():
    x = torch.randn(1, 2, 4, 5)

    conv = PackedDeformConv2d(2, 4,
                              kernel_size=(2, 3),
                              stride=(2, 3),
                              padding=(2, 3),
                              groups=2,
                              modulated=True)
    conv = torch.jit.script(conv)
    print(conv)

    out = conv(x)
    print(out.shape)


if __name__ == '__main__':
    test_wrapper()
    test_packed_wrapper()
