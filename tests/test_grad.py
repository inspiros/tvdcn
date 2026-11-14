import pytest
import torch
import torch.nn.functional as F

from .utils.cases import (PROFILES, forward, make_case, operator, output_and_grads)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.float16], ids=['double', 'float', 'half'])
def test_operator_paths(dim, transposed, profile, dtype, device):
    case = make_case(dim, transposed, profile, device=device, dtype=dtype)
    output, grads = output_and_grads(case)
    assert tuple(output.shape) == case['output_shape']
    assert output.dtype == dtype
    assert output.device.type == device
    assert torch.isfinite(output).all()
    for name, grad in grads.items():
        assert grad.shape == case['tensors'][name].shape
        assert torch.isfinite(grad).all(), name


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('profile', ['plain', 'offset', 'mask', 'both'])
def test_gradcheck(dim, transposed, profile, device):
    case = make_case(dim, transposed, profile, device=device, small=True)
    names = [name for name, value in case['tensors'].items() if value is not None]

    def call(*values):
        tensors = dict(case['tensors'])
        tensors.update(zip(names, values))
        return forward(case, tensors)

    assert torch.autograd.gradcheck(call, tuple(case['tensors'][name] for name in names),
                                    fast_mode=True, nondet_tol=1e-8)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('profile', ['plain', 'grouped', 'geometry'])
def test_ordinary_convolution_parity(dim, transposed, profile, device):
    case = make_case(dim, transposed, profile, device=device)
    tensors = case['tensors']
    # Zero offsets and unit masks must reduce to the corresponding torch op.
    for name, fill in [('offset', 0), ('mask', 1)]:
        if tensors[name] is not None:
            tensors[name] = torch.full_like(tensors[name], fill, requires_grad=True)
    actual = forward(case)
    reference_op = getattr(F, f"conv{'_transpose' if transposed else ''}{dim}d")
    expected = reference_op(tensors['input'], tensors['weight'], tensors['bias'], **case['kwargs'])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    params = tuple(tensors[name] for name in ('input', 'weight', 'bias') if tensors[name] is not None)
    probe = torch.linspace(-1, 1, actual.numel(), device=device, dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad(actual, params, probe)
    expected_grads = torch.autograd.grad(expected, params, probe)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
def test_int_arguments(dim, transposed, device):
    # Scalar stride/padding/dilation must be expanded like torch.nn.functional.
    case = make_case(dim, transposed, 'plain', device=device)
    tensors = case['tensors']
    kwargs = dict(stride=1, padding=0, dilation=1)
    if transposed:
        kwargs['output_padding'] = 0
    with torch.no_grad():
        actual = operator(case)(tensors['input'], tensors['weight'], bias=tensors['bias'], **kwargs)
        reference_op = getattr(F, f"conv{'_transpose' if transposed else ''}{dim}d")
        expected = reference_op(tensors['input'], tensors['weight'], tensors['bias'], **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
