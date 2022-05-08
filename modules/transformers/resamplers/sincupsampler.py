from modules.transformers import Transformer
from modules.transformers.convolution import StatefulConvolution

import numpy as np
from scipy import signal

class SincUpsampler(Transformer):
    def __init__(self, up=8, ntaps=49, window='hann') -> None:
        super().__init__()
        self.up = up

        taps = np.sinc(np.arange(-ntaps // 2 + 1, ntaps // 2 + 1, dtype=float) / up)
        taps *= signal.get_window(window, ntaps)
        self.conv = StatefulConvolution(taps)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        x = np.zeros(len(data) * self.up)
        x[::self.up] = data
        return self.conv(x)
