import pytest
import torch
import torchvision

from .utils.cases import (make_case, output_and_grads)


@pytest.mark.parametrize('profile', ['offset', 'both', 'grouped'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32], ids=['double', 'float'])
def test_torchvision_parity(profile, dtype, device):
    case = make_case(2, False, profile, dtype=dtype, device=device)
    if profile == 'grouped':
        # Torchvision requires mask_groups == offset_groups.
        case['tensors']['mask'] = case['tensors']['mask'].detach().repeat(1, 2, 1, 1).requires_grad_()
    actual, actual_grads = output_and_grads(case)
    tensors = {name: value.detach().clone().requires_grad_() if value is not None else None
               for name, value in case['tensors'].items()}
    kwargs = {key: value for key, value in case['kwargs'].items() if key != 'groups'}
    expected = torchvision.ops.deform_conv2d(tensors['input'], tensors['offset'], tensors['weight'],
                                            tensors['bias'], mask=tensors['mask'], **kwargs)
    probe = torch.linspace(-0.7, 0.9, expected.numel(), device=device, dtype=dtype).reshape_as(expected)
    names = [name for name, value in tensors.items() if value is not None]
    expected_grads = torch.autograd.grad(expected, tuple(tensors[name] for name in names), probe)
    tolerance = dict(rtol=1e-10, atol=1e-12) if dtype == torch.float64 else dict(rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    for name, grad in zip(names, expected_grads):
        torch.testing.assert_close(actual_grads[name], grad, **tolerance)
