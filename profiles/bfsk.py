from typing import List, Literal

from modules.transformers import Sequential
from modules.transformers.demodulators import BFSKDemodulator
from modules.transformers.filters import DCBlocker

from .common import get_cdr_deframer_block, get_preprocessing_block

def get_pipeline(
    fs: float,
    sps: int,
    carrier_freqs: List[float] = [3000.0, 3200.0],
    carrier_f_delta: float = 100.0,
    frame_format: Literal["standard", "payload_no_ecc", "payload_no_ecc_lc"] = "standard"
):
    assert len(carrier_freqs) == 2, "BFSK only supports two carrier frequencies"

    return Sequential(
        get_preprocessing_block(
            fs=fs,
            carrier_freqs=carrier_freqs,
            carrier_f_delta=carrier_f_delta,
        ),
        BFSKDemodulator(
            fs=fs,
            f0=carrier_freqs[0],
            f1=carrier_freqs[1],
            f_delta=carrier_f_delta,
            carrier_bandpass_ntaps=1229,
            symbol_lpf_cutoff_freq=1100,
            symbol_lpf_ntaps=405,
        ),
        get_cdr_deframer_block(
            sps=sps,
            cdr_type="simple",
            frame_format=frame_format,
        )
    )
