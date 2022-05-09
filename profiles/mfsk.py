from typing import List, Literal

from modules.transformers import Sequential
from modules.transformers.demodulators import MFSKDemodulator
from modules.transformers.filters import DCBlocker
from modules.transformers.preprocessing import GeneralPreprocessor

from .common import get_cdr_deframer_block

import math

def get_pipeline(
    fs: float,
    sps: int,
    carrier_freqs: List[float] = [3000.0, 3200.0, 3600.0, 3400.0],
    carrier_f_delta: float = 100.0,
    frame_format: Literal["standard", "raw_payload"] = "standard",
    frame_ecc_level: float = 0.2,
):
    n_freqs = len(carrier_freqs)
    assert math.log2(n_freqs).is_integer() and n_freqs >= 2, "MFSK only supports power-of-two and at-least-two carrier frequencies"

    return Sequential(
        GeneralPreprocessor(
            fs=fs,
            carrier_freqs=carrier_freqs,
            carrier_f_delta=carrier_f_delta,
            det_snr_threshold_db=6.0,
        ),
        MFSKDemodulator(
            fs=fs,
            freqs=carrier_freqs,
            f_delta=carrier_f_delta,
            carrier_bandpass_ntaps=1229,
            symbol_lpf_cutoff_freq=1100,
            symbol_lpf_ntaps=405,
        ),
        get_cdr_deframer_block(
            sps=sps,
            cdr_type="multicarrier",
            n_ary=int(math.log2(n_freqs)),
            frame_format=frame_format,
            frame_ecc_level=frame_ecc_level,
        )
    )
