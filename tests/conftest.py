import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--device', choices=('cpu', 'cuda', 'all'), default='cpu',
                     help='Devices on which to run operator tests (default: cpu).')
    parser.addoption('--run-packaging', action='store_true', default=False,
                     help='Run slow extension-build and packaging checks.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: builds native extensions or wheels')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-packaging'):
        skip = pytest.mark.skip(reason='requires --run-packaging')
        for item in items:
            if item.get_closest_marker('slow'):
                item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if 'device' in metafunc.fixturenames:
        requested = metafunc.config.getoption('--device')
        devices = ['cpu', 'cuda'] if requested == 'all' else [requested]
        metafunc.parametrize('device', devices)


@pytest.fixture(autouse=True)
def device_availability(request):
    callspec = getattr(request.node, 'callspec', None)
    if callspec is not None and callspec.params.get('device') == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA requested but unavailable')


@pytest.fixture(autouse=True, scope='session')
def single_cpu_thread():
    # Tiny cases run much faster without a thread-pool launch per operation.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
