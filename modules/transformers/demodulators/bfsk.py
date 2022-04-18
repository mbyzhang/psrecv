from modules.transformers.demodulators.am import AMDemodulator
from modules import transformers

import numpy as np

class BFSKDemodulator(transformers.Transformer):
    def __init__(
        self, 
        fs=48000,
        f0=3200,
        f1=3000, 
        f_delta=100, 
        carrier_bandpass_ntaps=1229, 
        symbol_lpf_cutoff_freq=1100,
        symbol_lpf_ntaps=405,
    ):
        self.am_demod_f0 = AMDemodulator(
            fs=fs,
            carrier_bandpass_cutoff=[f0 - f_delta, f0 + f_delta],
            carrier_bandpass_ntaps=carrier_bandpass_ntaps,
            lowpass_cutoff=symbol_lpf_cutoff_freq,
            lowpass_ntaps=symbol_lpf_ntaps
        )

        self.am_demod_f1 = AMDemodulator(
            fs=fs,
            carrier_bandpass_cutoff=[f1 - f_delta, f1 + f_delta],
            carrier_bandpass_ntaps=carrier_bandpass_ntaps,
            lowpass_cutoff=symbol_lpf_cutoff_freq,
            lowpass_ntaps=symbol_lpf_ntaps
        )

        self.frag_f0_amp = None
        self.frag_f1_amp = None
        self.frag_f1_f0_diff = None

    def __call__(self, fragment: np.ndarray) -> np.ndarray:
        frag_f0_amp = self.am_demod_f0(fragment)
        frag_f1_amp = self.am_demod_f1(fragment)
        frag_f1_f0_diff = frag_f1_amp - frag_f0_amp

        # for debugging
        self.frag_f0_amp = frag_f0_amp
        self.frag_f1_amp = frag_f1_amp
        self.frag_f1_f0_diff = frag_f1_f0_diff

        return frag_f1_f0_diff
