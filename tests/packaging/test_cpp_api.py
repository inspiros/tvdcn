"""Build a standalone stable C++ consumer of the public convolution wrappers."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from torch.utils.cpp_extension import include_paths, library_paths

from tvdcn.extension import _get_extension_path


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='standalone C++ smoke currently uses GNU linker flags')
def test_cpp_api(tmp_path):
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / 'cpp_api_smoke'
    libdirs = library_paths()
    command = [os.environ.get('CXX', 'c++'), '-std=c++20', '-O2',
               '-DTORCH_TARGET_VERSION=0x020e000000000000',
               '-I' + str(root / 'tvdcn/csrc'),
               *('-I' + p for p in include_paths()),
               str(Path(__file__).with_name('cpp_api') / 'smoke.cpp'),
               '-Wl,--no-as-needed', _get_extension_path('_C'), '-Wl,--as-needed',
               *('-L' + p for p in libdirs), '-ltorch', '-ltorch_cpu', '-lc10',
               *('-Wl,-rpath,' + p for p in libdirs),
               '-Wl,-rpath,' + str(root / 'tvdcn'), '-o', str(binary)]
    build = subprocess.run(command, text=True, capture_output=True)
    assert build.returncode == 0, build.stdout + build.stderr
    env = {**os.environ, 'LD_LIBRARY_PATH': os.pathsep.join(
        [*libdirs, os.environ.get('LD_LIBRARY_PATH', '')])}
    run = subprocess.run([str(binary)], text=True, capture_output=True, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'CPU smoke passed' in run.stdout
