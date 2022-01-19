from convolution import StatefulConvolver
from scipy import signal

import numpy as np

class Demodulator():
    def __init__(
        self, 
        fs=48000,
        f1=3000, 
        f2=3200, 
        f_delta=100, 
        carrier_bandpass_ntaps=1229, 
        symbol_lpf_cutoff_freq=1100,
        symbol_lpf_ntaps=405,
        eps=1e-6,
    ):
        f1_taps = signal.firwin(carrier_bandpass_ntaps, [f1 - f_delta, f1 + f_delta], fs=fs, pass_zero=False)
        f2_taps = signal.firwin(carrier_bandpass_ntaps, [f2 - f_delta, f2 + f_delta], fs=fs, pass_zero=False)
        # bg_taps = signal.firwin(carrier_bandpass_ntaps, [f1 - 5 * f_delta, f1 - 2 * f_delta, f2 + 2 * f_delta, f2 + 5 * f_delta], fs=fs, pass_zero=False)

        self.f1_bpf = StatefulConvolver(f1_taps)
        self.f2_bpf = StatefulConvolver(f2_taps)
        # self.bg_bpf = StatefulConvolver(bg_taps)

        symbol_lpf_taps = signal.firwin(symbol_lpf_ntaps, symbol_lpf_cutoff_freq, fs=fs)
        self.f1_symbol_lpf = StatefulConvolver(symbol_lpf_taps)
        self.f2_symbol_lpf = StatefulConvolver(symbol_lpf_taps)

        self.frag_f1_amp = None
        self.frag_f2_amp = None

        self.eps = eps
    
    def accept(self, fragment: np.ndarray) -> np.ndarray:
        frag_f1_sig = self.f1_bpf.convolve(fragment)
        frag_f2_sig = self.f2_bpf.convolve(fragment)
        # frag_bg_sig = self.bg_bpf.convolve(fragment)
        
        frag_f1_amp = self.f1_symbol_lpf.convolve(np.abs(frag_f1_sig) ** 2)
        frag_f2_amp = self.f2_symbol_lpf.convolve(np.abs(frag_f2_sig) ** 2)
        # frag_bg_amp = self.bg_symbol_lpf.convolve(np.abs(frag_bg_sig) ** 2)

        # frag_f1_active = (frag_f1_amp / frag_bg_amp > f_rela_threshold) & (frag_f1_amp > f_abs_threshold)
        # frag_f2_active = (frag_f2_amp / frag_bg_amp > f_rela_threshold) & (frag_f2_amp > f_abs_threshold)
        # frag_active = frag_f1_active | frag_f2_active
        
        # symbols = np.where(f1_active, 1, 0) + np.where(f2_active, -1, 0)

        # bg_amp_db = np.log10(bg_amp) * 10.0
        # f1_snr_db = np.log10(f1_amp) * 10.0 - bg_amp_db
        # f2_snr_db = np.log10(f2_amp) * 10.0 - bg_amp_db

        # for debugging
        self.frag_f1_amp = frag_f1_amp
        self.frag_f2_amp = frag_f2_amp

        eps = 1e-6 
        # It seems adding a eps to the signal can also solve the noise problem when there are no transmission signal. 
        # But this may have problems if the transmission signal is small
        frag_f1_f2_ratios = (np.log10(frag_f1_amp + eps) - np.log10(frag_f2_amp + eps)) * 10.0 # * signal.convolve(frag_f1_active | frag_f2_active, sig_det_lpf_taps, mode="same")
        
        return frag_f1_f2_ratios
