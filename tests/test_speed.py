import argparse
from collections import namedtuple

import torch
import torchvision
from tqdm import trange

import tvdcn
from utils.time_meter import TimeMeter

TimeMetersPair = namedtuple('ForwardBackwardTimeMetersPair',
                            ['forward', 'backward'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deformable', type=bool, default=True)
    parser.add_argument('--modulated', type=bool, default=True)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-n', '--n', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(1)
    input = torch.randn(1, 4, 32, 32, device=device, requires_grad=True)
    weight = torch.randn(2, 2, 5, 5, device=device, requires_grad=True)
    offset = torch.randn(1, 50, 28, 13, device=device, requires_grad=True) if args.deformable else None
    mask = torch.randn(1, 25, 28, 13, device=device, requires_grad=True) if args.modulated else None
    bias = torch.randn(2, device=device, requires_grad=True) if args.bias else None

    stride = (1, 2)
    padding = (0, 1)
    dilation = (1, 2)
    groups = 2

    params = [input, weight]
    if args.deformable:
        params.append(offset)
    if args.modulated:
        params.append(mask)
    if args.bias:
        params.append(bias)
    optim = torch.optim.Optimizer(params, {})

    # aliasing
    torchvision_f = torchvision.ops.deform_conv2d
    torchvision_args = (input, offset, weight, bias, stride, padding, dilation, mask)
    tvdcn_f = tvdcn.ops.deform_conv2d
    tvdcn_args = (input, weight, offset, mask, bias, stride, padding, dilation, groups)

    # warmup
    output = torchvision_f(*torchvision_args) + tvdcn_f(*tvdcn_args)
    grad = torch.ones_like(output)
    output.backward(grad)
    optim.zero_grad()

    torchvision_time_meter = TimeMetersPair(TimeMeter(), TimeMeter())
    for iter_id in (pbar := trange(args.n, desc=torchvision_f.__name__)):
        optim.zero_grad()
        with torchvision_time_meter.forward:
            output = torchvision_f(*torchvision_args)
        with torchvision_time_meter.backward:
            output.backward(grad)
        pbar.set_description(f'[{torchvision_f.__module__}.{torchvision_f.__name__} - iter {iter_id + 1}], '
                             f'forward_fps={torchvision_time_meter.forward.fps:.04f}, '
                             f'backward_fps={torchvision_time_meter.backward.fps:.04f}')

    tvdcn_time_meter = TimeMetersPair(TimeMeter(), TimeMeter())
    for iter_id in (pbar := trange(args.n, desc=tvdcn_f.__name__)):
        optim.zero_grad()
        with tvdcn_time_meter.forward:
            output = tvdcn_f(*tvdcn_args)
        with tvdcn_time_meter.backward:
            output.backward(grad)
        pbar.set_description(f'[{tvdcn_f.__module__}.{tvdcn_f.__name__} - iter {iter_id + 1}], '
                             f'forward_fps={tvdcn_time_meter.forward.fps:.04f}, '
                             f'backward_fps={tvdcn_time_meter.backward.fps:.04f}')


if __name__ == '__main__':
    main()
