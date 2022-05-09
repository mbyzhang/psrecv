from dsp.filters import FIRFilter
from dsp import DSPBlock
from scipy import signal

import numpy as np

class AMDemodulator(DSPBlock):
    def __init__(
        self, 
        fs,
        carrier_bandpass_cutoff,
        carrier_bandpass_ntaps=999,
        lowpass_cutoff=1000,
        lowpass_ntaps=499,
    ) -> None:
        super().__init__()

        self.carrier_filter = FIRFilter(
            fs=fs,
            cutoff=carrier_bandpass_cutoff,
            ntaps=carrier_bandpass_ntaps,
            pass_zero=False
        )

        self.lowpass_filter = FIRFilter(
            fs=fs,
            cutoff=lowpass_cutoff,
            ntaps=lowpass_ntaps,
            pass_zero=True
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        carrier_signal = self.carrier_filter(data)
        amplitudes = self.lowpass_filter(np.abs(carrier_signal) ** 2)
        return amplitudes
