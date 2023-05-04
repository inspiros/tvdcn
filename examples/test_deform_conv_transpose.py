import torch

from tvdcn import *


def test_wrapper():
    conv = DeformConvTranspose2d(4, 2, kernel_size=(2, 2))
    conv = torch.jit.script(conv)
    print(conv)


def test_packed_wrapper():
    x = torch.randn(1, 4, 4, 5)

    conv = PackedDeformConvTranspose2d(4, 2,
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
