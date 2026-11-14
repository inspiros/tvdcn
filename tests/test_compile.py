"""Dispatcher, compiler, export and dtype coverage for the stable operator families."""
import itertools
import sys

import pytest
import torch
import tvdcn

from .utils.cases import PROFILES, forward, make_case, output_and_grads


pytestmark = [pytest.mark.parametrize('dim', [1, 2, 3]),
              pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])]


def _args(case):
    keys = ['stride', 'padding']
    if case['transposed']:
        keys.append('output_padding')
    keys += ['dilation', 'groups']
    return (*case['tensors'].values(), *(case['kwargs'][key] for key in keys))


def _op_name(dim, transposed):
    return f"deform_conv{'_transpose' if transposed else ''}{dim}d"


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('bias', [True, False], ids=['bias', 'no-bias'])
def test_opcheck(dim, transposed, profile, bias, dtype, device):
    case = make_case(dim, transposed, profile, device=device, dtype=dtype)
    if not bias:
        case['tensors']['bias'] = None
    torch.library.opcheck(getattr(torch.ops.tvdcn, _op_name(dim, transposed)).default, _args(case))
    grad = torch.ones(case['output_shape'], device=device, dtype=dtype)
    # This also checks defined empty outputs for absent optional gradients.
    torch.library.opcheck(getattr(torch.ops.tvdcn, f'_{_op_name(dim, transposed)}_backward').default,
                          (grad, *_args(case)),
                          test_utils=('test_schema', 'test_faketensor'))


# torch 2.14 CUDA builds crash (access violation) in torch.cuda.current_device() on
# Windows machines without a GPU, which Inductor calls while hashing the graph.
@pytest.mark.skipif(sys.platform == 'win32' and torch.version.cuda is not None
                    and not torch.cuda.is_available(),
                    reason='torch CUDA init crashes on GPU-less Windows')
@pytest.mark.parametrize('profile', ['plain', 'offset', 'mask', 'both', 'geometry'])
def test_compile_forward_backward(dim, transposed, profile, device):
    case = make_case(dim, transposed, profile, device=device, dtype=torch.float32)
    names = [k for k, v in case['tensors'].items() if v is not None]
    values = tuple(case['tensors'][k] for k in names)

    def call(*args):
        tensors = dict(case['tensors'])
        tensors.update(zip(names, args))
        return forward(case, tensors)

    # `call` shares one code object across parametrizations; reset so the
    # CPU+CUDA grid does not exceed Dynamo's recompile limit.
    torch._dynamo.reset()
    compiled = torch.compile(call, fullgraph=True)
    expected = call(*values)
    actual = compiled(*values)
    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(-0.7, 0.9, expected.numel(), device=device).reshape_as(expected)
    expected_grads = torch.autograd.grad(expected, values, probe)
    actual_grads = torch.autograd.grad(actual, values, probe)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_export(dim, transposed, device):
    case = make_case(dim, transposed, 'geometry', device=device, dtype=torch.float32)
    module = getattr(tvdcn, f"DeformConv{'Transpose' if transposed else ''}{dim}d")(
        2, 4, 2, device=device, **case['kwargs'])
    args = tuple(case['tensors'][k] for k in ('input', 'offset', 'mask'))
    exported = torch.export.export(module, args)
    torch.testing.assert_close(exported.module()(*args), module(*args))


def test_dispatcher_defaults(dim, transposed, device):
    case = make_case(dim, transposed, 'plain', device=device)
    # Exercise elided trailing defaults in the dispatcher and autograd context.
    torch.library.opcheck(getattr(torch.ops.tvdcn, _op_name(dim, transposed)).default,
                          (case['tensors']['input'], case['tensors']['weight']))


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16], ids=['bf16', 'half'])
def test_low_precision(dim, transposed, profile, dtype, device):
    reference = make_case(dim, transposed, profile, device=device, dtype=torch.float32)
    case = make_case(dim, transposed, profile, device=device, dtype=dtype)
    # Compare against float32 evaluated at exactly the same quantized inputs.
    reference['tensors'] = {k: v.detach().float().requires_grad_() if v is not None else None
                            for k, v in case['tensors'].items()}
    actual, grads = output_and_grads(case)
    expected, expected_grads = output_and_grads(reference)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.006)
    for name, grad in grads.items():
        assert grad.dtype == dtype
        torch.testing.assert_close(grad.float(), expected_grads[name], rtol=0.06, atol=0.012)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('profile', ['plain', 'both'])
def test_autocast(dim, transposed, profile, dtype, device):
    case = make_case(dim, transposed, profile, device=device, dtype=dtype)
    # Autocast computes in float32 (float64 stays double), then casts back.
    ref = dict(case)
    ref['tensors'] = {k: (v.detach().to(torch.float64 if dtype == torch.float64 else torch.float32)
                          .requires_grad_()) if v is not None else None
                      for k, v in case['tensors'].items()}
    expected, expected_grads = output_and_grads(ref)
    with torch.autocast(device, dtype=torch.bfloat16 if device == 'cpu' else torch.float16):
        actual, grads = output_and_grads(case)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected.to(dtype))
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name].to(dtype), rtol=0.02, atol=0.002)


@pytest.mark.parametrize('optional', list(itertools.product([False, True], repeat=3)))
def test_optional_needs_grad_and_unbatched(dim, transposed, optional, device):
    case = make_case(dim, transposed, device=device)
    for name, enabled in zip(('offset', 'mask', 'bias'), optional):
        if not enabled:
            case['tensors'][name] = None
    # Only input requires grad. Optional present tensors still must be saved.
    for name, value in case['tensors'].items():
        if value is not None:
            if name != 'bias':
                value = value.detach().squeeze(0)
            case['tensors'][name] = value.detach().requires_grad_(name == 'input')
    output = forward(case)
    output.sum().backward()
    assert case['tensors']['input'].grad.shape == case['tensors']['input'].shape


def test_double_backward_rejected(dim, transposed, device):
    case = make_case(dim, transposed, device=device)
    value = case['tensors']['input']
    first, = torch.autograd.grad(forward(case).sum(), value, create_graph=True)
    with pytest.raises(RuntimeError, match='double backwards'):
        torch.autograd.grad(first.sum(), value)


@pytest.mark.parametrize('unbatched', [False, True])
def test_opcheck_noncontiguous_single_channel(dim, transposed, unbatched, device):
    case = make_case(dim, transposed, device=device)
    tensors = case['tensors']
    for name in ('input', 'weight'):
        value = tensors[name][:1] if transposed and name == 'weight' else tensors[name][:, :1]
        value = (value.repeat_interleave(2, dim=-1)[..., ::2] if dim == 1
                 else value.transpose(-1, -2))
        assert not value.is_contiguous()
        tensors[name] = value.detach().requires_grad_()
    if unbatched:
        for name in ('input', 'offset', 'mask'):
            tensors[name] = tensors[name].squeeze(0).detach().requires_grad_()
    torch.library.opcheck(getattr(torch.ops.tvdcn, _op_name(dim, transposed)).default, _args(case))
