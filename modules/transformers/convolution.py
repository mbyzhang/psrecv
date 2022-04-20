from scipy import signal
from modules.transformers import Transformer
import numpy as np

class StatefulConvolution(Transformer):
    def __init__(self, fir_filter):
        super().__init__()
        self.fir_filter = fir_filter
        self.ntaps = len(fir_filter)
        self.padding = np.zeros(self.ntaps - 1)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.convolve(data)

    def convolve(self, sample):
        x = np.concatenate((self.padding, sample))
        self.padding = x[-self.ntaps + 1:]
        return signal.convolve(x, self.fir_filter, mode='valid')
