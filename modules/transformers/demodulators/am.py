from modules.transformers.convolution import StatefulConvolution
from scipy import signal

import numpy as np
from modules import transformers

class AMDemodulator(transformers.Transformer):
    def __init__(
        self, 
        fs,
        carrier_bandpass_cutoff,
        carrier_bandpass_ntaps=999,
        lowpass_cutoff=1000,
        lowpass_ntaps=499,
    ) -> None:
        super().__init__()

        carrier_taps = signal.firwin(carrier_bandpass_ntaps, carrier_bandpass_cutoff, fs=fs, pass_zero=False)
        lowpass_taps = signal.firwin(lowpass_ntaps, lowpass_cutoff, fs=fs)

        self.carrier_filter = StatefulConvolution(carrier_taps)
        self.lowpass_filter = StatefulConvolution(lowpass_taps)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        carrier_signal = self.carrier_filter(data)
        amplitudes = self.lowpass_filter(np.abs(carrier_signal) ** 2)
        return amplitudes
