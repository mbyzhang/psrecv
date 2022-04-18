from modules.transformers import Sequential
from modules.transformers.cdr import SimpleCDR
from modules.transformers.framing import Deframer
from modules.transformers.demodulators import BFSKDemodulator

get_pipeline = lambda fs, sps: \
    Sequential(
        BFSKDemodulator(
            fs=fs,
            f0=3000,
            f1=3200,
            f_delta=100,
            carrier_bandpass_ntaps=1229,
            symbol_lpf_cutoff_freq=1100,
            symbol_lpf_ntaps=405,
        ),
        SimpleCDR(
            sps=sps,
            clk_recovery_window=sps // 4,
            clk_recovery_grad_threshold=1e-8,
            median_window_size=int(sps * 0.8)
        ),
        Deframer(
            format=Deframer.FormatType.STANDARD
        )
    )
