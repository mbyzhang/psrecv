class Transformer():
    def accept(self, data = None):
        raise NotImplementedError()

from .sequential import Sequential
from .bitunpacking import BitUnpacking
