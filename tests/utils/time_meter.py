import time

__all__ = ['TimeMeter']


class TimeMeter:
    __slots__ = 'elapsed_time', 'n', '_ts', '_te'

    def __init__(self):
        self.elapsed_time = self.n = 0
        self._ts = self._te = None

    def __enter__(self):
        self._ts = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._te = time.time()
        self.n += 1
        self.elapsed_time += self._te - self._ts

    def reset(self):
        self.elapsed_time = self.n = 0
        self._ts = self._te = None

    @property
    def fps(self):
        return self.n / self.elapsed_time if self.elapsed_time else float('nan')
