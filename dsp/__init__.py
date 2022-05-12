class DSPBlock():
    def __call__(self, data = None):
        raise NotImplementedError()

from .sequential import Sequential
from .bitunpacking import BitUnpacking
from .statscollector import StatisticsCollector
from .identity import Identity
from .delay import Delay
from .diffdecoder import DifferentialDecoder
