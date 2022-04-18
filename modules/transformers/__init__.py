class Transformer():
    def __call__(self, data = None):
        raise NotImplementedError()

from .sequential import Sequential
from .bitunpacking import BitUnpacking
