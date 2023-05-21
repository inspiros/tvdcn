import torch
from torch.autograd.gradcheck import gradcheck

import tvdcn
from utils.deform_conv_test_args import DeformConvTestArgs


def test_deform_conv(dim=2,
                     dtype=torch.float64,
                     device='cuda'):
    torch.manual_seed(12)
    conv_func = getattr(tvdcn, f'deform_conv{dim}d')
    args = DeformConvTestArgs(dim=dim, transposed=False, dtype=dtype, device=device)
    print(args)

    c_output = conv_func(args.input,
                         args.weight,
                         args.offset,
                         args.mask,
                         args.bias,
                         args.stride,
                         args.padding,
                         args.dilation,
                         args.groups)
    c_output.sum().backward()
    c_input_grad = args.input.grad.clone()
    c_weight_grad = args.weight.grad.clone()
    c_offset_grad = args.offset.grad.clone()
    c_mask_grad = args.mask.grad.clone()
    c_bias_grad = args.bias.grad.clone()
    args.zero_grad()
    print(c_output)

    grad_ok = gradcheck(
        lambda inp, wei, off, msk, bi: conv_func(inp, wei, off, msk, bi,
                                                 args.stride,
                                                 args.padding,
                                                 args.dilation,
                                                 args.groups),
        (args.input, args.weight, args.offset, args.mask, args.bias), nondet_tol=args.tol)
    args.zero_grad()
    print('grad_check:', grad_ok)


def test_deform_conv_transpose(dim=2,
                               dtype=torch.float64,
                               device='cuda'):
    torch.manual_seed(12)
    conv_func = getattr(tvdcn, f'deform_conv_transpose{dim}d')
    args = DeformConvTestArgs(dim=dim, transposed=True, dtype=dtype, device=device)
    print(args)

    c_output = conv_func(args.input,
                         args.weight,
                         args.offset,
                         args.mask,
                         args.bias,
                         args.stride,
                         args.padding,
                         args.output_padding,
                         args.dilation,
                         args.groups)
    c_output.sum().backward()
    c_input_grad = args.input.grad.clone()
    c_weight_grad = args.weight.grad.clone()
    c_offset_grad = args.offset.grad.clone()
    c_mask_grad = args.mask.grad.clone()
    c_bias_grad = args.bias.grad.clone()
    args.zero_grad()
    print(c_output)

    grad_ok = gradcheck(
        lambda inp, wei, off, msk, bi: conv_func(inp, wei, off, msk, bi,
                                                 args.stride,
                                                 args.padding,
                                                 args.output_padding,
                                                 args.dilation,
                                                 args.groups),
        (args.input, args.weight, args.offset, args.mask, args.bias), nondet_tol=args.tol)
    args.zero_grad()
    print('grad_check:', grad_ok)


if __name__ == '__main__':
    test_deform_conv()
    test_deform_conv_transpose()
