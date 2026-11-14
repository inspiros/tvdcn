"""Enable real opcheck/compile/export coverage after the stable ABI port (WP3)."""
import pytest


@pytest.mark.xfail(reason='WP3: legacy C++ autograd has no torch.library fake/compile support', strict=True)
def test_compile_and_opcheck_pending():
    pytest.fail('Replace this placeholder with opcheck, compile and export tests in WP3')
