import torch
import torchvision

import tvdcn
from utils.deform_conv_test_args import DeformConvTestArgs


def test_deform_conv2d(dtype=torch.float64,
                       device='cuda',
                       with_mask=True):
    torch.manual_seed(12)
    args = DeformConvTestArgs(dim=2,
                              offset_groups=1,
                              mask_groups=1,
                              dtype=dtype,
                              device=device)
    print(args)

    args.zero_grad()
    tvdcn_output = tvdcn.ops.deform_conv2d(args.input,
                                           args.weight,
                                           args.offset,
                                           args.mask if with_mask else None,
                                           args.bias,
                                           args.stride,
                                           args.padding,
                                           args.dilation,
                                           args.groups)
    tvdcn_output.sum().backward()
    tvdcn_input_grad = args.input.grad.clone()
    tvdcn_weight_grad = args.weight.grad.clone()
    tvdcn_offset_grad = args.offset.grad.clone()
    tvdcn_mask_grad = args.mask.grad.clone() if with_mask else torch.zeros([0])
    tvdcn_bias_grad = args.bias.grad.clone()

    args.zero_grad()
    torchvision_output = torchvision.ops.deform_conv2d(args.input,
                                                       args.offset,
                                                       args.weight,
                                                       args.bias,
                                                       args.stride,
                                                       args.padding,
                                                       args.dilation,
                                                       args.mask if with_mask else None)
    torchvision_output.sum().backward()
    torchvision_input_grad = args.input.grad.clone()
    torchvision_weight_grad = args.weight.grad.clone()
    torchvision_offset_grad = args.offset.grad.clone()
    torchvision_mask_grad = args.mask.grad.clone() if with_mask else torch.zeros([0])
    torchvision_bias_grad = args.bias.grad.clone()

    print('output_diff:', tvdcn_output - torchvision_output, sep='\n')
    assert torch.allclose(tvdcn_output, torchvision_output)

    print('\ngrad_diff:')
    print(f'\tinput_grad:         {torch.allclose(tvdcn_input_grad, torchvision_input_grad)}')
    assert torch.allclose(tvdcn_input_grad, torchvision_input_grad)
    print(f'\tweight_grad:        {torch.allclose(tvdcn_weight_grad, torchvision_weight_grad)}')
    assert torch.allclose(tvdcn_weight_grad, torchvision_weight_grad)
    print(f'\toffset_grad:        {torch.allclose(tvdcn_offset_grad, torchvision_offset_grad)}')
    assert torch.allclose(tvdcn_offset_grad, torchvision_offset_grad)
    print(f'\tmask_grad:          {torch.allclose(tvdcn_mask_grad, torchvision_mask_grad)}')
    assert torch.allclose(tvdcn_mask_grad, torchvision_mask_grad)
    print(f'\tbias_grad:          {torch.allclose(tvdcn_bias_grad, torchvision_bias_grad)}')
    assert torch.allclose(tvdcn_bias_grad, torchvision_bias_grad)


if __name__ == '__main__':
    test_deform_conv2d()
