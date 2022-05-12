from dsp import DSPBlock, Delay
from dsp.filters import FIRFilter
from dsp.resamplers import Decimator, SincUpsampler
from dsp.pll import CostasLoop

import numpy as np

class DBPSKDemodulator(DSPBlock):
    def __init__(
        self,
        fs,
        sps,
        f,
        f_delta=100,
        carrier_bandpass_ntaps=1229,
        costas_lpf_ntaps=405,
        resampler_up=8,
    ) -> None:
        super().__init__()
        fs_r = fs * resampler_up
        self.resampler_up = resampler_up

        self.upsampler = SincUpsampler(up=resampler_up)

        self.carrier_filter = FIRFilter(
            fs=fs,
            cutoff=[f - f_delta, f + f_delta],
            ntaps=carrier_bandpass_ntaps,
            pass_zero=False
        )

        self.loop = CostasLoop(
            f=f,
            fs=fs_r,
            update_period=int(fs_r / f * 5),
            alpha=0.2,
            beta=1e-7,
            lpf_cutoff=f / 2.0,
            lpf_ntaps=costas_lpf_ntaps,
        )

        self.decimator = Decimator(down=resampler_up)

        self.symbol_delay = Delay(sps)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        carrier = self.carrier_filter(data)
        carrier_up = self.upsampler(carrier)
        phase_abs_up = self.loop(carrier_up)
        phase_abs = self.decimator(phase_abs_up)

        self.stat_carrier = carrier
        self.stat_phase_abs = phase_abs

        return phase_abs
