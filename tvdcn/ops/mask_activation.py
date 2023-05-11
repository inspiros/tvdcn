import torch
from torch.jit.annotations import Tuple
from torch.nn.modules.utils import _single, _pair, _triple

__all__ = [
    'mask_softmax1d',
    'mask_softmax2d',
    'mask_softmax3d',
]


def mask_softmax1d(mask: torch.Tensor,
                   kernel_size: Tuple[int]) -> torch.Tensor:
    r"""
    Performs 1D Mask Softmax Normalization.

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_width, out_width]): modulation
            masks to be multiplied with each output of convolution kernel.
        kernel_size (int or Tuple[int]): convolution kernel size.
    """
    batch_size, _, out_width = mask.size()

    weight_w, = _single(kernel_size)
    mask_groups = mask.size(1) // weight_w

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0].\n"
            "Got mask.shape[1]={}, while kernel_size[0]={}".format(
                mask.shape[1], weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_w,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_w,
                     out_width)
    return mask


def mask_softmax2d(mask: torch.Tensor,
                   kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""
    Performs 2D Mask Softmax Normalization.

    References:
        https://arxiv.org/abs/2211.05778

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_height * kernel_width,
            out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel.
        kernel_size (int or Tuple[int, int]): convolution kernel size.
    """
    batch_size, _, out_height, out_width = mask.size()

    weight_h, weight_w = _pair(kernel_size)
    mask_groups = mask.size(1) // (weight_h * weight_w)

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0] * kernel_size[1].\n"
            "Got mask.shape[1]={}, while kernel_size[0] * kernel_size[1]={}".format(
                mask.shape[1], weight_h * weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_h * weight_w,
                     out_height,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_h * weight_w,
                     out_height,
                     out_width)
    return mask


def mask_softmax3d(mask: torch.Tensor,
                   kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""
    Performs 3D Mask Softmax Normalization.

    Arguments:
        mask (Tensor[batch_size, mask_groups * kernel_depth * kernel_height * kernel_width,
            out_depth, out_height, out_width]): modulation masks to be multiplied with each output
            of convolution kernel. Default: None
        kernel_size (int or Tuple[int, int]): convolution kernel size.
    """
    batch_size, _, out_depth, out_height, out_width = mask.size()

    weight_d, weight_h, weight_w = _triple(kernel_size)
    mask_groups = mask.size(1) // (weight_d * weight_h * weight_w)

    if mask_groups == 0:
        raise RuntimeError(
            "The shape of the mask tensor at dimension 1 is not valid. It should "
            "be a multiple of kernel_size[0] * kernel_size[1] * kernel_size[2].\n"
            "Got mask.shape[1]={}, while kernel_size[0] * kernel_size[1] * kernel_size[2]={}".format(
                mask.shape[1], weight_d * weight_h * weight_w))

    mask = mask.view(batch_size,
                     mask_groups,
                     weight_d * weight_h * weight_w,
                     out_depth,
                     out_height,
                     out_width)
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(batch_size,
                     mask_groups * weight_d * weight_h * weight_w,
                     out_depth,
                     out_height,
                     out_width)
    return mask
