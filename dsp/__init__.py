class DSPBlock():
    def __call__(self, data = None):
        raise NotImplementedError()

from .sequential import Sequential
from .bitunpacking import BitUnpacking
from .auxdatacollector import AuxDataCollector
from .identity import Identity
from .delay import Delay
