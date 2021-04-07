import math
import unittest
from functools import lru_cache

import torch
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair

import tvdcn as ops


class OpTester(object):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def test_forward_cpu_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=True)

    def test_forward_cpu_non_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=False)

    def test_backward_cpu_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=True)

    def test_backward_cpu_non_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_non_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_non_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=False)

    def _test_forward(self, device, contiguous):
        pass

    def _test_backward(self, device, contiguous):
        pass


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


class DeformConvTester(OpTester, unittest.TestCase):
    def expected_fn(self, x, weight, offset, mask, bias, stride=1, padding=0, dilation=1):
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)
        weight_h, weight_w = weight.shape[-2:]

        n_batches, n_in_channels, in_h, in_w = x.shape
        n_out_channels = weight.shape[0]

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        n_offset_grps = offset.shape[1] // (2 * weight_h * weight_w)
        in_c_per_offset_grp = n_in_channels // n_offset_grps

        n_weight_grps = n_in_channels // weight.shape[1]
        in_c_per_weight_grp = weight.shape[1]
        out_c_per_weight_grp = n_out_channels // n_weight_grps

        out = torch.zeros(n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        for b in range(n_batches):
            for c_out in range(n_out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for di in range(weight_h):
                            for dj in range(weight_w):
                                for c in range(in_c_per_weight_grp):
                                    weight_grp = c_out // out_c_per_weight_grp
                                    c_in = weight_grp * in_c_per_weight_grp + c

                                    offset_grp = c_in // in_c_per_offset_grp
                                    mask_idx = offset_grp * (weight_h * weight_w) + di * weight_w + dj
                                    offset_idx = 2 * mask_idx

                                    pi = stride_h * i - pad_h + dil_h * di + offset[b, offset_idx, i, j]
                                    pj = stride_w * j - pad_w + dil_w * dj + offset[b, offset_idx + 1, i, j]

                                    mask_value = 1.0
                                    if mask is not None:
                                        mask_value = mask[b, mask_idx, i, j]

                                    out[b, c_out, i, j] += (mask_value * weight[c_out, c, di, dj] *
                                                            bilinear_interpolate(x[b, c_in, :, :], pi, pj))
        out += bias.view(1, n_out_channels, 1, 1)
        return out

    @lru_cache(maxsize=None)
    def get_fn_args(self, device, contiguous, batch_sz, dtype):
        n_in_channels = 6
        n_out_channels = 2
        n_weight_grps = 2
        n_offset_grps = 3

        stride = (2, 1)
        pad = (1, 0)
        dilation = (2, 1)

        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (3, 2)
        in_h, in_w = (5, 4)

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        x = torch.rand(batch_sz, n_in_channels, in_h, in_w, device=device, dtype=dtype, requires_grad=True)

        offset = torch.randn(batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w,
                             device=device, dtype=dtype, requires_grad=True)

        mask = torch.randn(batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w,
                           device=device, dtype=dtype, requires_grad=True)

        weight = torch.randn(n_out_channels, n_in_channels // n_weight_grps, weight_h, weight_w,
                             device=device, dtype=dtype, requires_grad=True)

        bias = torch.randn(n_out_channels, device=device, dtype=dtype, requires_grad=True)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            mask = mask.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        return x, weight, offset, mask, bias, stride, pad, dilation

    def _test_forward(self, device, contiguous, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        for batch_sz in [0, 33]:
            self._test_forward_with_batchsize(device, contiguous, batch_sz, dtype)

    def _test_forward_with_batchsize(self, device, contiguous, batch_sz, dtype):
        x, _, offset, mask, _, stride, padding, dilation = self.get_fn_args(device, contiguous, batch_sz, dtype)
        in_channels = 6
        out_channels = 2
        kernel_size = (3, 2)
        groups = 2
        tol = 2e-3 if dtype is torch.half else 1e-5

        layer = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups).to(device=x.device, dtype=dtype)
        res = layer(x, offset, mask)

        weight = layer.weight.data
        bias = layer.bias.data
        expected = self.expected_fn(x, weight, offset, mask, bias, stride=stride, padding=padding, dilation=dilation)

        self.assertTrue(torch.allclose(res.to(expected.dtype), expected, rtol=tol, atol=tol),
                        '\nres:\n{}\nexpected:\n{}'.format(res, expected))

        # no modulation test
        res = layer(x, offset)
        expected = self.expected_fn(x, weight, offset, None, bias, stride=stride, padding=padding, dilation=dilation)

        self.assertTrue(torch.allclose(res.to(expected.dtype), expected, rtol=tol, atol=tol),
                        '\nres:\n{}\nexpected:\n{}'.format(res, expected))

        # test for wrong sizes
        with self.assertRaises(RuntimeError):
            wrong_offset = torch.rand_like(offset[:, :2])
            res = layer(x, wrong_offset)

        with self.assertRaises(RuntimeError):
            wrong_mask = torch.rand_like(mask[:, :2])
            res = layer(x, offset, wrong_mask)

    def _test_backward(self, device, contiguous):
        for batch_sz in [0, 33]:
            self._test_backward_with_batchsize(device, contiguous, batch_sz)

    def _test_backward_with_batchsize(self, device, contiguous, batch_sz):
        x, weight, offset, mask, bias, stride, padding, dilation = self.get_fn_args(device, contiguous,
                                                                                    batch_sz, self.dtype)

        def func(x_, offset_, mask_, weight_, bias_):
            return ops.deform_conv2d(x_, weight_, offset_, mask_, bias_, stride=stride,
                                     padding=padding, dilation=dilation)

        gradcheck(func, (x, offset, mask, weight, bias), nondet_tol=1e-5)

        def func_no_mask(x_, offset_, weight_, bias_):
            return ops.deform_conv2d(x_, weight_, offset_, None, bias_, stride=stride,
                                     padding=padding, dilation=dilation)

        gradcheck(func_no_mask, (x, offset, weight, bias), nondet_tol=1e-5)

        @torch.jit.script
        def script_func(x_, offset_, mask_, weight_, bias_, stride_, pad_, dilation_):
            # type:(Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int])->Tensor
            return ops.deform_conv2d(x_, weight_, offset_, mask_, bias_, stride=stride_,
                                     padding=pad_, dilation=dilation_)

        gradcheck(lambda z, off, msk, wei, bi: script_func(z, off, msk, wei, bi, stride, padding, dilation),
                  (x, offset, mask, weight, bias), nondet_tol=1e-5)

        @torch.jit.script
        def script_func_no_mask(x_, offset_, weight_, bias_, stride_, pad_, dilation_):
            # type:(Tensor, Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int])->Tensor
            return ops.deform_conv2d(x_, weight_, offset_, None, bias_, stride=stride_,
                                     padding=pad_, dilation=dilation_)

        gradcheck(lambda z, off, wei, bi: script_func_no_mask(z, off, wei, bi, stride, padding, dilation),
                  (x, offset, weight, bias), nondet_tol=1e-5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_compare_cpu_cuda_grads(self):
        # Test from https://github.com/pytorch/vision/issues/2598
        # Run on CUDA only
        for contiguous in [False, True]:
            # compare grads computed on CUDA with grads computed on CPU
            true_cpu_grads = None

            init_weight = torch.randn(9, 9, 3, 3, requires_grad=True)
            img = torch.randn(8, 9, 1000, 110)
            offset = torch.rand(8, 2 * 3 * 3, 1000, 110)
            mask = torch.rand(8, 3 * 3, 1000, 110)

            if not contiguous:
                img = img.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
                offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
                mask = mask.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
                weight = init_weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)
            else:
                weight = init_weight

            for d in ["cpu", "cuda"]:

                out = ops.deform_conv2d(img.to(d), weight.to(d), offset.to(d), mask.to(d), padding=1)
                out.mean().backward()
                if true_cpu_grads is None:
                    true_cpu_grads = init_weight.grad
                    self.assertTrue(true_cpu_grads is not None)
                else:
                    self.assertTrue(init_weight.grad is not None)
                    res_grads = init_weight.grad.to("cpu")
                    self.assertTrue(true_cpu_grads.allclose(res_grads))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_autocast(self):
        for dtype in (torch.float, torch.half):
            with torch.cuda.amp.autocast():
                self._test_forward(torch.device("cuda"), False, dtype=dtype)


if __name__ == '__main__':
    unittest.main()
