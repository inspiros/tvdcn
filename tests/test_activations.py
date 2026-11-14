import math

import pytest
import torch
from tvdcn.ops import activations


@pytest.mark.parametrize('scale', [0.5, 1.0, 2.0])
def test_mask_sigmoid(scale, device):
    x = torch.linspace(-3, 3, 12, dtype=torch.float64, device=device).requires_grad_()
    expected = scale * x.sigmoid()
    actual = activations.mask_sigmoid(x, scale)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(activations.MaskSigmoid(scale)(x), expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               scale * x.sigmoid() * (1 - x.sigmoid()))
    torch.testing.assert_close(activations.mask_sigmoid(torch.zeros_like(x)), torch.ones_like(x))


@pytest.mark.parametrize('scale', [0, -1])
def test_mask_sigmoid_invalid_scale(scale, device):
    with pytest.raises(ValueError, match='positive'):
        activations.mask_sigmoid(torch.zeros(1, device=device), scale)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2])
def test_mask_softmax(dim, groups, device):
    kernel = tuple(range(2, 2 + dim))
    volume = math.prod(kernel)
    shape = (1, groups * volume, *((2,) * dim))
    x = torch.linspace(-2, 2, math.prod(shape), device=device, dtype=torch.float64).reshape(shape).requires_grad_()
    fn = getattr(activations, f'mask_softmax{dim}d')
    module = getattr(activations, f'MaskSoftmax{dim}d')(kernel)
    expected = x.reshape(1, groups, volume, *((2,) * dim)).softmax(2).reshape(shape)
    actual = fn(x, kernel)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module(x), expected)
    torch.testing.assert_close(module(x, kernel_size=kernel), expected)
    torch.testing.assert_close(actual.reshape(1, groups, volume, -1).sum(2),
                               torch.ones(1, groups, 2 ** dim, dtype=x.dtype, device=device))
    assert torch.autograd.gradcheck(lambda value: fn(value, kernel), (x,), fast_mode=True)
    with pytest.raises(RuntimeError):
        fn(torch.zeros(1, volume - 1, *((2,) * dim), device=device), kernel)
