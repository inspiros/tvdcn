import torch
from torch.autograd.gradcheck import gradcheck

import tvdcn
from utils.deform_conv_test_args import DeformConvTestArgs


def test_deform_conv(dim=1):
    torch.manual_seed(12)
    conv_func = torch.jit.script(getattr(tvdcn, f'deform_conv{dim}d'))
    args = DeformConvTestArgs(dim=dim, device='cuda:0', dtype=torch.float64, batch_size=1)
    print(args)

    c_res = conv_func(args.input,
                      args.weight,
                      args.offset,
                      args.mask,
                      args.bias,
                      args.stride,
                      args.padding,
                      args.dilation,
                      args.weight_groups)
    c_res.sum().backward()
    c_input_grad = args.input.grad.clone()
    c_weight_grad = args.weight.grad.clone()
    c_offset_grad = args.offset.grad.clone()
    c_mask_grad = args.mask.grad.clone()
    c_bias_grad = args.bias.grad.clone()
    args.zero_grad()
    print(c_res)
    # print(c_input_grad)
    # torch.save(c_res, "D:/Script/libs/tvdcn/test_data/c_res.pt")
    # torch.save(c_input_grad, "D:/Script/libs/tvdcn/test_data/c_input_grad.pt")
    # torch.save(c_weight_grad, "D:/Script/libs/tvdcn/test_data/c_weight_grad.pt")
    # torch.save(c_offset_grad, "D:/Script/libs/tvdcn/test_data/c_offset_grad.pt")
    # torch.save(c_mask_grad, "D:/Script/libs/tvdcn/test_data/c_mask_grad.pt")
    # torch.save(c_bias_grad, "D:/Script/libs/tvdcn/test_data/c_bias_grad.pt")

    grad_ok = gradcheck(
        lambda inp, wei, off, msk, bi: conv_func(inp, wei, off, msk, bi,
                                                 args.stride,
                                                 args.padding,
                                                 args.dilation,
                                                 args.weight_groups),
        (args.input, args.weight, args.offset, args.mask, args.bias), nondet_tol=args.tol)
    print('grad_check:', grad_ok)


if __name__ == '__main__':
    test_deform_conv()
