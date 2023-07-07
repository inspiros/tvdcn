import torch
import torch.nn as nn
from collections import OrderedDict
from tvdcn import PackedDeformConvTranspose2d


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, name='VGG16', num_classes=10):
        super(VGG, self).__init__()

        cur_cfg = cfg[name]
        self.features = self._make_layers(cur_cfg)
        last_conv_out_channels = self.features[-3].out_channels
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(last_conv_out_channels, cur_cfg[-1])),
                    ("norm1", nn.BatchNorm1d(cur_cfg[-1])),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(cur_cfg[-1], num_classes)),
                ]
            )
        )

    def _make_layers(self, cfg):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0

        for i, x in enumerate(cfg):
            if x == "M":
                layers.add_module("pool%d" % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                cnt += 1
                layer = PackedDeformConvTranspose2d(
                    in_channels,
                    x,
                    kernel_size=3,
                    padding=1,
                    modulated=True,
                )
                layers.add_module("conv%d" % i, layer)
                layers.add_module("norm%d" % i, nn.BatchNorm2d(x))
                layers.add_module("relu%d" % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def vgg11():

    return VGG(name='VGG11')


def vgg13():

    return VGG(name='VGG13')


def vgg16():

    return VGG(name='VGG16')


def vgg19():

    return VGG(name='VGG19')
