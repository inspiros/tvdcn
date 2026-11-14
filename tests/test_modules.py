import math

import pytest
import torch
import torch.nn.functional as F
import tvdcn

from .utils.cases import forward, make_case, mark_legacy_optional_backward


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('profile', ['plain', 'grouped', 'geometry'])
def test_module_forward_and_reset(dim, transposed, profile, device):
    case = make_case(dim, transposed, profile, device=device)
    cls = getattr(tvdcn, f"DeformConv{'Transpose' if transposed else ''}{dim}d")
    module = cls(2, 2, 2, bias=case['tensors']['bias'] is not None,
                 device=device, dtype=torch.float64, **case['kwargs'])
    with torch.no_grad():
        module.weight.copy_(case['tensors']['weight'])
        if module.bias is not None:
            module.bias.copy_(case['tensors']['bias'])
    actual = module(case['tensors']['input'], case['tensors']['offset'], case['tensors']['mask'])
    torch.testing.assert_close(actual, forward(case))
    assert tuple(actual.shape) == case['output_shape']
    before = module.weight.detach().clone()
    module.reset_parameters()
    assert not torch.equal(before, module.weight)
    bound = 1 / math.sqrt((2 // case['kwargs']['groups']) * 2 ** dim)
    assert module.weight.abs().max() <= bound
    if module.bias is not None:
        assert module.bias.abs().max() <= bound


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True], ids=['conv', 'transpose'])
@pytest.mark.parametrize('deformable,modulated', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('activation', ['sigmoid', 'softmax'])
def test_packed_modules(dim, transposed, deformable, modulated, activation, device, request):
    mark_legacy_optional_backward(request, not (deformable and modulated))
    cls = getattr(tvdcn, f"PackedDeformConv{'Transpose' if transposed else ''}{dim}d")
    kwargs = dict(stride=2, padding=1, dilation=2, groups=2)
    if transposed:
        kwargs['output_padding'] = 1
    # Fork RNG avoids coupling later cases to module initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        module = cls(2, 2, 2, offset_groups=2, mask_groups=2, deformable=deformable,
                     modulated=modulated, mask_activation=activation, generator_bias=True,
                     device=device, dtype=torch.float64, **kwargs)
    x = make_case(dim, transposed, device=device)['tensors']['input']
    expected_op = getattr(F, f"conv{'_transpose' if transposed else ''}{dim}d")
    # Generators start at zero: offsets=0; masks=1 (sigmoid) or 1/kernel_volume.
    scale = 1 / 2 ** dim if modulated and activation == 'softmax' else 1
    expected = expected_op(x, module.weight * scale, module.bias, **kwargs)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    actual.square().sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    before = module.weight.detach().clone()
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(2)
    module.reset_parameters()
    assert not torch.equal(before, module.weight)
    for generator in (module.offset_generator, module.mask_generator):
        if generator is not None:
            for param in generator.parameters():
                assert torch.count_nonzero(param) == 0


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_transposed_output_size(dim, device):
    module = getattr(tvdcn, f'DeformConvTranspose{dim}d')(2, 2, 2, stride=2,
                                                                       device=device, dtype=torch.float64)
    x = make_case(dim, True, device=device)['tensors']['input']
    shape = (1, 2, *((7,) * dim))
    assert module(x, None, output_size=shape).shape == shape
    with pytest.raises(ValueError, match='valid sizes'):
        module(x, None, output_size=(9,) * dim)
