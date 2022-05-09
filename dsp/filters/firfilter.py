from dsp.convolution import StatefulConvolution
from dsp import DSPBlock
from scipy import signal

import numpy as np

class FIRFilter(DSPBlock):
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
