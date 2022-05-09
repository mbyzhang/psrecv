from typing import List, Literal

from modules.transformers import Sequential
from modules.transformers.demodulators import DBPSKDemodulator
from modules.transformers.preprocessing import GeneralPreprocessor

from .common import get_cdr_deframer_block

def get_pipeline(
    fs: float,
    sps: int,
    carrier_freqs: List[float] = [3000.0],
    carrier_f_delta: float = 100.0,
    frame_format: Literal["standard", "raw_payload"] = "standard",
    frame_ecc_level: float = 0.2,
):
    assert len(carrier_freqs) == 1, "MFSK only supports one carrier frequency"

    return Sequential(
        GeneralPreprocessor(
            fs=fs,
            carrier_freqs=carrier_freqs,
            carrier_f_delta=carrier_f_delta,
            det_snr_threshold_db=6.0,
        ),
        DBPSKDemodulator(
            fs=fs,
            sps=sps,
            f=carrier_freqs[0],
            f_delta=carrier_f_delta,
            resampler_up=8,
        ),
        get_cdr_deframer_block(
            sps=sps,
            cdr_type="simple",
            frame_format=frame_format,
            frame_ecc_level=frame_ecc_level,
        )
    )
