import torch
from torch import Tensor
import torch.nn as nn

__all__ = [
    'mask_sigmoid',
    'MaskSigmoid',
]


def mask_sigmoid(mask: Tensor, scale: float = 2.0) -> Tensor:
    r"""
    Applies scaled sigmoid activation on mask. The default scale is set to 2 so that initial
    values when ``conv_mask`` is initialized with zeros is 1.

    References:
        https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py

    Args:
        mask (Tensor[batch_size, mask_groups * kernel_area, *out_shape]): modulation masks
            to be multiplied with each output of convolution kernel.
        scale (float): positive scaling of the activation output.
    """
    if scale <= 0:
        raise ValueError('scale must be positive. Got scale={}'.format(scale))
    return torch.sigmoid(mask) * scale


class MaskSigmoid(nn.Module):
    """
    See :func:`mask_sigmoid`.
    """

    def __init__(self, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, mask: Tensor) -> Tensor:
        return mask_sigmoid(mask, self.scale)
