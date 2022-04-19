from modules.transformers.convolution import StatefulConvolution
from modules import transformers
from scipy import signal

import numpy as np

class FIRFilter(transformers.Transformer):
    def __init__(
        self,
        fs,
        cutoff,
        ntaps=999,
        pass_zero=True
    ) -> None:
        super().__init__()
        taps = signal.firwin(ntaps, cutoff, fs=fs, pass_zero=pass_zero)
        self.filter = StatefulConvolution(taps)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.filter(data)
