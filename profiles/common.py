from typing import List, Literal

from modules.transformers import Sequential, BitUnpacking, Transformer
from modules.transformers.cdr import SimpleCDR, MulticarrierCDR
from modules.transformers.framing import Deframer
from modules.transformers.demodulators import BFSKDemodulator
from modules.transformers.preprocessing import CarrierDetectorExtractor, LogLoopAGC
from modules.transformers.filters import DCBlocker

def get_preprocessing_block(
    fs,
    carrier_freqs: List[float],
    carrier_f_delta=100.0
) -> Transformer:
    return Sequential(
        CarrierDetectorExtractor(
            fs=fs,
            carrier_cutoff=[min(carrier_freqs) - carrier_f_delta, max(carrier_freqs) + carrier_f_delta],
            noise_delta=carrier_f_delta,
            ntaps=1229,
            ema_alpha=0.6,
            snr_threshold_db=10,
            update_period=512,
        ),
        LogLoopAGC(
            step_size=0.2,
            ref=-3.0,
            gain_max=50.0,
            update_period=512,
        ),
    )

def get_cdr_deframer_block(
    sps,
    cdr_type: Literal["simple", "multicarrier"] = "simple",
    n_ary: int = 2,
    frame_format: Literal["standard", "payload_no_ecc", "payload_no_ecc_lc"] = "standard"
) -> Transformer:
    try:
        frame_format_enum = {
            "standard": Deframer.FormatType.STANDARD,
            "payload_no_ecc": None,
            "payload_no_ecc_lc": Deframer.FormatType.RAW_PAYLOAD,
        }[frame_format]
    except KeyError:
        raise ValueError(f"Unsupported frame format: {frame_format}")

    if cdr_type == "simple":
        return Sequential(
            SimpleCDR(
                sps=sps,
                clk_recovery_window=sps // 4,
                clk_recovery_grad_threshold=1e-3,
                median_window_size=int(sps * 0.8)
            ),
            Deframer(
                format=frame_format_enum
            )
        )
    elif cdr_type == "multicarrier":
        return Sequential(
            MulticarrierCDR(
                sps=sps,
                clk_recovery_window=sps // 4,
                clk_recovery_grad_threshold=1e-3,
                median_window_size=int(sps * 0.8)
            ),
            BitUnpacking(
                count=n_ary
            ),
            Deframer(
                format=frame_format_enum
            )
        )
    else:
        raise ValueError(f"Unsupported CDR type: {cdr_type}")
