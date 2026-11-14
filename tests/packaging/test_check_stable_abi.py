import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'tools/packaging/check_stable_abi.py'
pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    sys.platform != 'linux', reason='ABI inspection requires Linux binutils')]


def run(*args, cwd=None):
    result = subprocess.run([str(arg) for arg in args], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'MAX_JOBS': '2'})
    return result


@pytest.fixture(scope='module')
def stable_extension(tmp_path_factory):
    directory = tmp_path_factory.mktemp('stable-abi-build')
    shutil.copytree(Path(__file__).with_name('stable_ext'), directory, dirs_exist_ok=True)
    result = run(sys.executable, 'setup.py', 'bdist_wheel', cwd=directory)
    assert result.returncode == 0, result.stdout
    return directory, next((directory / 'dist').glob('*.whl'))


def test_stable_extension_passes(stable_extension):
    directory, wheel = stable_extension
    assert '-cp310-abi3-' in wheel.name
    module = next(directory.glob('build/lib*/_stable_ext*.so'))
    for artifact in (module, wheel):
        result = run(sys.executable, CHECKER, artifact)
        assert result.returncode == 0, result.stdout
        assert 'PASS' in result.stdout
    result = run(sys.executable, '-c',
                 'import torch, _stable_ext; '
                 'assert torch.ops.tvdcn_packaging_test.increment(41) == 42', cwd=module.parent)
    assert result.returncode == 0, result.stdout


def test_legacy_extension_fails():
    candidates = [ROOT / 'tvdcn' / ('_C' + suffix)
                  for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    module = next((path for path in candidates if path.exists()), None)
    assert module is not None, 'Build the legacy extension with setup.py build_ext --inplace first'
    result = run(sys.executable, CHECKER, module)
    assert result.returncode == 1, result.stdout
    assert 'FAIL' in result.stdout
    assert 'unstable torch C++ symbol' in result.stdout


def test_checker_checks_every_wheel_module(stable_extension, tmp_path):
    # Include one forbidden C++ import and one forbidden CPython import. The
    # stable module must not hide violations in the second module of a wheel.
    _, stable_wheel = stable_extension
    source = tmp_path / 'bad.cpp'
    source.write_text('extern "C" void _PyEval_EvalFrameDefault();\n'
                      'namespace c10 { void forbidden(); }\n'
                      'namespace at { void exported() {} }\n'
                      'extern "C" void probe() { c10::forbidden(); _PyEval_EvalFrameDefault(); }\n')
    # A synthetic dependency exercises DT_NEEDED independently of whichever
    # libraries the host linker retains for the real legacy extension.
    dependency = tmp_path / 'libtorch_python.so'
    result = run('c++', '-shared', '-fPIC', '-x', 'c++', '/dev/null', '-o', dependency)
    assert result.returncode == 0, result.stdout
    bad_module = tmp_path / '_bad.so'
    result = run('c++', '-shared', '-fPIC', source, '-o', bad_module,
                 '-Wl,--no-as-needed', f'-L{tmp_path}', '-ltorch_python')
    assert result.returncode == 0, result.stdout
    wheel = tmp_path / 'mixed.whl'
    with zipfile.ZipFile(stable_wheel) as original, zipfile.ZipFile(wheel, 'w') as mixed:
        for name in original.namelist():
            mixed.writestr(name, original.read(name))
        mixed.write(bad_module, 'package/_bad.so')
    result = run(sys.executable, CHECKER, wheel)
    assert result.returncode == 1, result.stdout
    assert 'PASS' in result.stdout and 'package/_bad.so' in result.stdout
    assert 'c10::forbidden()' in result.stdout
    assert 'at::exported()' in result.stdout
    assert 'libtorch_python.so' in result.stdout
    assert '_PyEval_EvalFrameDefault' in result.stdout
