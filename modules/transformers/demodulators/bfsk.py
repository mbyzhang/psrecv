from modules.transformers.convolution import StatefulConvolution
from scipy import signal

import numpy as np
import modules.transformers as transformers

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
        eps=1e-6,
    ):
        f0_taps = signal.firwin(carrier_bandpass_ntaps, [f0 - f_delta, f0 + f_delta], fs=fs, pass_zero=False)
        f1_taps = signal.firwin(carrier_bandpass_ntaps, [f1 - f_delta, f1 + f_delta], fs=fs, pass_zero=False)
        # bg_taps = signal.firwin(carrier_bandpass_ntaps, [f0 - 5 * f_delta, f0 - 2 * f_delta, f1 + 2 * f_delta, f1 + 5 * f_delta], fs=fs, pass_zero=False)

        self.f0_bpf = StatefulConvolution(f0_taps)
        self.f1_bpf = StatefulConvolution(f1_taps)
        # self.bg_bpf = StatefulConvolver(bg_taps)

        symbol_lpf_taps = signal.firwin(symbol_lpf_ntaps, symbol_lpf_cutoff_freq, fs=fs)
        self.f0_symbol_lpf = StatefulConvolution(symbol_lpf_taps)
        self.f1_symbol_lpf = StatefulConvolution(symbol_lpf_taps)

        self.frag_f0_amp = None
        self.frag_f1_amp = None

        self.eps = eps
    
    def accept(self, fragment: np.ndarray) -> np.ndarray:
        frag_f0_sig = self.f0_bpf.convolve(fragment)
        frag_f1_sig = self.f1_bpf.convolve(fragment)
        # frag_bg_sig = self.bg_bpf.convolve(fragment)
        
        frag_f0_amp = self.f0_symbol_lpf.convolve(np.abs(frag_f0_sig) ** 2)
        frag_f1_amp = self.f1_symbol_lpf.convolve(np.abs(frag_f1_sig) ** 2)
        # frag_bg_amp = self.bg_symbol_lpf.convolve(np.abs(frag_bg_sig) ** 2)

        # frag_f0_active = (frag_f0_amp / frag_bg_amp > f_rela_threshold) & (frag_f0_amp > f_abs_threshold)
        # frag_f1_active = (frag_f1_amp / frag_bg_amp > f_rela_threshold) & (frag_f1_amp > f_abs_threshold)
        # frag_active = frag_f0_active | frag_f1_active
        
        # symbols = np.where(f0_active, 1, 0) + np.where(f1_active, -1, 0)

        # bg_amp_db = np.log10(bg_amp) * 10.0
        # f0_snr_db = np.log10(f0_amp) * 10.0 - bg_amp_db
        # f1_snr_db = np.log10(f1_amp) * 10.0 - bg_amp_db

        # for debugging
        self.frag_f0_amp = frag_f0_amp
        self.frag_f1_amp = frag_f1_amp

        eps = 1e-6 
        # It seems adding a eps to the signal can also solve the noise problem when there are no transmission signal. 
        # But this may have problems if the transmission signal is small
        frag_f1_f0_ratios = (np.log10(frag_f1_amp + eps) - np.log10(frag_f0_amp + eps)) * 10.0 # * signal.convolve(frag_f0_active | frag_f1_active, sig_det_lpf_taps, mode="same")
        
        return frag_f1_f0_ratios
