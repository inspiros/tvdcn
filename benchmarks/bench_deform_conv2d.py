"""Small speed comparison of tvdcn.ops.deform_conv2d against torchvision.ops.deform_conv2d.

Usage: python benchmarks/bench_deform_conv2d.py [--device cuda|cpu] [--dtype float32|float16|float64]
Times forward and forward+backward with torch.utils.benchmark (median, CUDA-synchronized)
on a few typical shapes, after checking that both libraries produce the same output.
"""
import argparse

import torch
import torch.utils.benchmark as benchmark
import torchvision

import tvdcn

# (name, batch, in_channels, out_channels, height=width, kernel, offset_groups)
CONFIGS = [
    ('small', 8, 32, 32, 32, 3, 1),
    ('medium', 8, 64, 64, 64, 3, 1),
    ('large', 4, 128, 128, 128, 3, 4),
]


def make_inputs(batch, cin, cout, size, k, offset_groups, device, dtype):
    g = torch.Generator(device='cpu').manual_seed(0)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype).requires_grad_()

    out = size  # stride 1, padding k // 2
    return dict(
        input=rand(batch, cin, size, size),
        weight=rand(cout, cin, k, k, scale=0.1),
        offset=rand(batch, 2 * offset_groups * k * k, out, out),
        mask=torch.sigmoid(rand(batch, offset_groups * k * k, out, out)).detach().requires_grad_(),
        bias=rand(cout),
    )


def tvdcn_fn(t, padding):
    return tvdcn.ops.deform_conv2d(t['input'], t['weight'], t['offset'], t['mask'], t['bias'], padding=padding)


def torchvision_fn(t, padding):
    return torchvision.ops.deform_conv2d(t['input'], t['offset'], t['weight'], t['bias'],
                                         padding=padding, mask=t['mask'])


def fwd_bwd(fn, t, padding):
    out = fn(t, padding)
    torch.autograd.grad(out, list(t.values()), torch.ones_like(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'float64'])
    parser.add_argument('--min-run-time', type=float, default=1.0)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    device_name = torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'
    print(f'torch {torch.__version__}, torchvision {torchvision.__version__}, device {device_name}, {args.dtype}, '
          f'threads {torch.get_num_threads()}')
    print(f'{"config":8} {"pass":8} {"tvdcn ms":>10} {"torchvision ms":>15} {"ratio":>7}')
    for name, *shape in CONFIGS:
        t = make_inputs(*shape, device=args.device, dtype=dtype)
        padding = shape[4] // 2
        with torch.no_grad():
            torch.testing.assert_close(tvdcn_fn(t, padding), torchvision_fn(t, padding),
                                       rtol=1e-2 if dtype == torch.float16 else 1e-4,
                                       atol=1e-2 if dtype == torch.float16 else 1e-4)
        for label, stmt in (('forward', 'fn(t, padding)'), ('fwd+bwd', 'fwd_bwd(fn, t, padding)')):
            times = {}
            for lib, fn in (('tvdcn', tvdcn_fn), ('torchvision', torchvision_fn)):
                ctx = torch.no_grad() if label == 'forward' else torch.enable_grad()
                with ctx:
                    timer = benchmark.Timer(stmt, globals=dict(fn=fn, t=t, padding=padding, fwd_bwd=fwd_bwd))
                    times[lib] = timer.blocked_autorange(min_run_time=args.min_run_time).median * 1e3
            print(f'{name:8} {label:8} {times["tvdcn"]:10.3f} {times["torchvision"]:15.3f} '
                  f'{times["tvdcn"] / times["torchvision"]:7.2f}')


if __name__ == '__main__':
    main()
