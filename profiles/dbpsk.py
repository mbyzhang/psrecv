from typing import List, Literal

from dsp import Sequential, DifferentialDecoder
from dsp.demodulators import DBPSKDemodulator
from dsp.preprocessing import GeneralPreprocessor
from dsp.framing import Deframer
from dsp.cdr import SimpleCDR


def get_pipeline(
    fs: float,
    sps: int,
    carrier_freqs: List[float] = [3000.0],
    carrier_f_delta: float = 100.0,
    frame_format: Literal["standard", "raw_payload"] = "standard",
    frame_ecc_level: float = 0.2,
):
    assert len(carrier_freqs) == 1, "MFSK only supports one carrier frequency"

    frame_format_enum = {
        "standard": Deframer.FormatType.STANDARD,
        "raw_payload": Deframer.FormatType.RAW_PAYLOAD,
    }[frame_format]

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
        SimpleCDR(
            sps=sps,
            clk_recovery_window=sps // 4,
            clk_recovery_grad_threshold=1e-3,
            median_window_size=int(sps * 0.3)
        ),
        DifferentialDecoder(),
        Deframer(
            format=frame_format_enum,
            payload_parity_len_ratio=frame_ecc_level,
        ),
    )
