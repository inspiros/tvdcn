import torch

from tvdcn import PackedDeformConv2d


def main():
    x = torch.randn(1, 3, 4, 4)
    conv = PackedDeformConv2d(3, 4, kernel_size=(2, 2), modulated=True)
    out = conv(x)
    print(out)


if __name__ == '__main__':
    main()
