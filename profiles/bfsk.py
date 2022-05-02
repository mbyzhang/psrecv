from typing import List, Literal

from modules.transformers import Sequential
from modules.transformers.demodulators import BFSKDemodulator
from modules.transformers.filters import DCBlocker
from modules.transformers.preprocessing import GeneralPreprocessor

from .common import get_cdr_deframer_block

def get_pipeline(
    fs: float,
    sps: int,
    carrier_freqs: List[float] = [3000.0, 3200.0],
    carrier_f_delta: float = 100.0,
    frame_format: Literal["standard", "payload_no_ecc", "payload_no_ecc_lc"] = "standard"
):
    assert len(carrier_freqs) == 2, "BFSK only supports two carrier frequencies"

    return Sequential(
        GeneralPreprocessor(
            fs=fs,
            carrier_freqs=carrier_freqs,
            carrier_f_delta=carrier_f_delta,
            det_snr_threshold_db=6.0,
        ),
        BFSKDemodulator(
            fs=fs,
            f0=carrier_freqs[0],
            f1=carrier_freqs[1],
            f_delta=carrier_f_delta,
            carrier_bandpass_ntaps=1229,
            symbol_lpf_cutoff_freq=1100,
            symbol_lpf_ntaps=405,
            mode="ratio",
        ),
        DCBlocker(
            p=0.9999,
            mode="iir"
        ),
        get_cdr_deframer_block(
            sps=sps,
            cdr_type="simple",
            frame_format=frame_format,
        )
    )
