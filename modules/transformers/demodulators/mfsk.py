from modules.transformers.demodulators.am import AMDemodulator
from modules.transformers import Transformer

import numpy as np

class MFSKDemodulator(Transformer):
    def __init__(
        self,
        fs=48000,
        freqs=[3000, 3200],
        f_delta=100,
        carrier_bandpass_ntaps=1229,
        symbol_lpf_cutoff_freq=1100,
        symbol_lpf_ntaps=405,
    ):
        self.am_demods = [
            AMDemodulator(
                fs=fs,
                carrier_bandpass_cutoff=[f - f_delta, f + f_delta],
                carrier_bandpass_ntaps=carrier_bandpass_ntaps,
                lowpass_cutoff=symbol_lpf_cutoff_freq,
                lowpass_ntaps=symbol_lpf_ntaps
            ) for f in freqs
        ]

        self.frag_fi_envelope = None
        self.frag_diff = None

    def __call__(self, fragment: np.ndarray) -> np.ndarray:
        frag_fi_envelope = np.array([
            am_demod(fragment) for am_demod in self.am_demods
        ], copy=False)

        # for debugging
        self.frag_fi_envelope = frag_fi_envelope

        diff = []
        for i, envelope in enumerate(frag_fi_envelope):
            diff.append(envelope - np.sum(np.delete(frag_fi_envelope, i, axis=0), axis=0))

        diff = np.array(diff, copy=False)

        self.frag_diff = diff
        return diff
