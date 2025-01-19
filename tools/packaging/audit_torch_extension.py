import sys

from auditwheel.main import main


LIBTORCH_LIBRARIES = [
    'libtorch.so',
    'libtorch_cpu.so',
    'libtorch_cuda.so',
    'libtorch_python.so',
    'libc10.so',
    'libc10_cuda.so',
]

try:
    from auditwheel.policy import _POLICIES as POLICIES

    for p in POLICIES:
        p['lib_whitelist'].extend(LIBTORCH_LIBRARIES)
except ImportError:
    for lib in LIBTORCH_LIBRARIES:
        sys.argv.append('--exclude')
        sys.argv.append(lib)


if __name__ == "__main__":
    sys.exit(main())
