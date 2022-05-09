from typing import Literal

from dsp import Sequential, BitUnpacking, DSPBlock
from dsp.cdr import SimpleCDR, MulticarrierCDR
from dsp.framing import Deframer

def get_cdr_deframer_block(
    sps,
    cdr_type: Literal["simple", "multicarrier"] = "simple",
    n_ary: int = 2,
    frame_format: Literal["standard", "raw_payload"] = "standard",
    frame_ecc_level: float = 0.2
) -> DSPBlock:
    try:
        frame_format_enum = {
            "standard": Deframer.FormatType.STANDARD,
            "raw_payload": Deframer.FormatType.RAW_PAYLOAD,
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
                format=frame_format_enum,
                payload_parity_len_ratio=frame_ecc_level,
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
                format=frame_format_enum,
                payload_parity_len_ratio=frame_ecc_level,
            )
        )
    else:
        raise ValueError(f"Unsupported CDR type: {cdr_type}")
