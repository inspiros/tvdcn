import unittest
from functools import lru_cache

import numpy as np
import torch
from torch.autograd.gradcheck import gradcheck

from tvdcn.deform_conv import *


def _decontiguous(x: torch.Tensor):
    perm_ind = torch.arange(x.ndim)
    while perm_ind.allclose(torch.arange(x.ndim)):
        perm_ind = torch.randperm(x.ndim)
    deperm_ind = torch.argsort(perm_ind)
    return x.permute(*perm_ind.tolist()).contiguous().permute(*deperm_ind.tolist())


class DeformConvTester(unittest.TestCase):
    @staticmethod
    @lru_cache(maxsize=None)
    def _get_fn_args(
            dim,
            device="cpu",
            contiguous=True,
            batch_sz=1,
            dtype=torch.float64
    ):
        n_in_channels = 6
        n_out_channels = 2
        n_weight_grps = 2
        n_offset_grps = 3

        stride = torch.arange(1, 1 + dim).flip(0).tolist()
        padding = torch.arange(0, 0 + dim).flip(0).tolist()
        dilation = torch.arange(1, 1 + dim).flip(0).tolist()

        kernel_size = torch.arange(2, 2 + dim).flip(0).tolist()
        in_size = torch.arange(4, 4 + dim).flip(0).tolist()
        out_size = [(in_size[_] + 2 * padding[_] - (dilation[_] * (kernel_size[_] - 1) + 1)) // stride[_] + 1
                    for _ in range(dim)]

        x = torch.rand(batch_sz, n_in_channels, *in_size, device=device, dtype=dtype, requires_grad=True)

        offset = torch.randn(batch_sz, n_offset_grps * dim * np.prod(kernel_size), *out_size,
                             device=device, dtype=dtype, requires_grad=True)

        mask = torch.randn(batch_sz, n_offset_grps * np.prod(kernel_size), *out_size,
                           device=device, dtype=dtype, requires_grad=True)

        weight = torch.randn(n_out_channels, n_in_channels // n_weight_grps, *kernel_size,
                             device=device, dtype=dtype, requires_grad=True)

        bias = torch.randn(n_out_channels, device=device, dtype=dtype, requires_grad=True)

        if not contiguous:
            x = _decontiguous(x)
            offset = _decontiguous(offset)
            mask = _decontiguous(mask)
            weight = _decontiguous(weight)

        if dim == 1:
            stride = stride[0]
            padding = padding[0]
            dilation = dilation[0]
        return (x, weight, offset, mask, bias,
                n_in_channels, n_out_channels, kernel_size, stride, padding, dilation)

    @staticmethod
    def _get_test_fn(dim):
        if dim == 1:
            return deform_conv1d
        elif dim == 2:
            return deform_conv2d
        elif dim == 3:
            return deform_conv3d
        raise RuntimeError(f'dim must be in [1, 2, 3], got {dim}')

    @staticmethod
    def _get_test_class(dim, packed):
        if dim == 1:
            return DeformConv1d if not packed else PackedDeformConv1d
        elif dim == 2:
            return DeformConv2d if not packed else PackedDeformConv2d
        elif dim == 3:
            return DeformConv3d if not packed else PackedDeformConv3d
        raise RuntimeError(f'dim must be in {1, 2, 3}, got {dim}')

    def _test_function_forward(self, dim, device, contiguous):
        dtype = torch.float64
        deform_func = self._get_test_fn(dim)
        (x, weight, offset, mask, bias, n_in_channels, n_out_channels,
         kernel_size, stride, padding, dilation) = self._get_fn_args(dim,
                                                                     device=device,
                                                                     contiguous=contiguous,
                                                                     batch_sz=1,
                                                                     dtype=dtype)

        deform_func(x, weight, offset, mask, bias, stride, padding, dilation)
        # TODO: check output

    def _test_function_backward(self, dim, device, contiguous):
        dtype = torch.float64
        deform_func = self._get_test_fn(dim)
        (x, weight, offset, mask, bias, n_in_channels, n_out_channels,
         kernel_size, stride, padding, dilation) = self._get_fn_args(dim,
                                                                     device=device,
                                                                     contiguous=contiguous,
                                                                     batch_sz=1,
                                                                     dtype=dtype)

        script_func = torch.jit.script(deform_func)
        gradcheck(lambda inp, wei, off, msk, bi: script_func(inp, wei, off, msk, bi, stride, padding, dilation),
                  (x, weight, offset, mask, bias), nondet_tol=1e-5)

    def _test_layers(self, device):
        dtype = torch.float64
        for dim in [1, 2, 3]:
            deform_conv_cls = self._get_test_class(dim, packed=False)
            (x, weight, offset, mask, bias, n_in_channels, n_out_channels,
             kernel_size, stride, padding, dilation) = self._get_fn_args(dim,
                                                                         device=device,
                                                                         contiguous=True,
                                                                         batch_sz=1,
                                                                         dtype=dtype)
            layer = deform_conv_cls(n_in_channels,
                                    n_out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=True).to(device=device, dtype=dtype)
            layer.weight.data = weight
            layer.bias.data = bias

            layer(x, offset, mask)
            # TODO: check output

    def _test_packed_layers(self, device):
        dtype = torch.float64
        for dim in [1, 2, 3]:
            deform_conv_cls = self._get_test_class(dim, packed=True)
            (x, weight, _, _, bias, n_in_channels, n_out_channels,
             kernel_size, stride, padding, dilation) = self._get_fn_args(dim,
                                                                         device=device,
                                                                         contiguous=True,
                                                                         batch_sz=1,
                                                                         dtype=torch.float64)
            layer = deform_conv_cls(n_in_channels,
                                    n_out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=True,
                                    modulated=True).to(device=device, dtype=dtype)
            layer.weight.data = weight
            layer.bias.data = bias

            layer(x)
            # TODO: check output

    def test_deform_conv1d_forward_cpu_contiguous(self):
        self._test_function_forward(dim=1, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv1d_forward_cpu_non_contiguous(self):
        self._test_function_forward(dim=1, device=torch.device('cpu'), contiguous=False)

    def test_deform_conv2d_forward_cpu_contiguous(self):
        self._test_function_forward(dim=2, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv2d_forward_cpu_non_contiguous(self):
        self._test_function_forward(dim=2, device=torch.device('cpu'), contiguous=False)

    def test_deform_conv3d_forward_cpu_contiguous(self):
        self._test_function_forward(dim=3, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv3d_forward_cpu_non_contiguous(self):
        self._test_function_forward(dim=3, device=torch.device('cpu'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv1d_forward_cuda_contiguous(self):
        self._test_function_forward(dim=1, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv1d_forward_cuda_non_contiguous(self):
        self._test_function_forward(dim=1, device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv2d_forward_cuda_contiguous(self):
        self._test_function_forward(dim=2, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv2d_forward_cuda_non_contiguous(self):
        self._test_function_forward(dim=2, device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv3d_forward_cuda_contiguous(self):
        self._test_function_forward(dim=3, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv3d_forward_cuda_non_contiguous(self):
        self._test_function_forward(dim=3, device=torch.device('cuda'), contiguous=False)

    def test_deform_conv1d_backward_cpu_contiguous(self):
        self._test_function_backward(dim=1, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv1d_backward_cpu_non_contiguous(self):
        self._test_function_backward(dim=1, device=torch.device('cpu'), contiguous=False)

    def test_deform_conv2d_backward_cpu_contiguous(self):
        self._test_function_backward(dim=2, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv2d_backward_cpu_non_contiguous(self):
        self._test_function_backward(dim=2, device=torch.device('cpu'), contiguous=False)

    def test_deform_conv3d_backward_cpu_contiguous(self):
        self._test_function_backward(dim=3, device=torch.device('cpu'), contiguous=True)

    def test_deform_conv3d_backward_cpu_non_contiguous(self):
        self._test_function_backward(dim=3, device=torch.device('cpu'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv1d_backward_cuda_contiguous(self):
        self._test_function_backward(dim=1, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv1d_backward_cuda_non_contiguous(self):
        self._test_function_backward(dim=1, device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv2d_backward_cuda_contiguous(self):
        self._test_function_backward(dim=2, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv2d_backward_cuda_non_contiguous(self):
        self._test_function_backward(dim=2, device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv3d_backward_cuda_contiguous(self):
        self._test_function_backward(dim=3, device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_deform_conv3d_backward_cuda_non_contiguous(self):
        self._test_function_backward(dim=3, device=torch.device('cuda'), contiguous=False)

    def test_cpu_layers(self):
        self._test_layers(torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_layers(self):
        self._test_layers(torch.device('cuda'))

    def test_cpu_packed_layers(self):
        self._test_packed_layers(torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_packed_layers(self):
        self._test_packed_layers(torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
