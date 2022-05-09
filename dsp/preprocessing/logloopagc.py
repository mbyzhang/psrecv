from typing import Optional
from dsp import DSPBlock

import numpy as np

# Logarithmic-Loop AGC
class LogLoopAGC(DSPBlock):
    def __init__(
        self,
        update_period = 512,
        gain_initial = 0.0,
        gain_max = 60.0,
        step_size = 0.2,
        decay_step_size = 0.2,
        ref = 0.0,
    ) -> None:
        super().__init__()
        self.update_period = update_period
        self.gain = gain_initial
        self.gain_max = gain_max
        self.ref = ref
        self.step_size = step_size
        self.decay_step_size = decay_step_size

    def __call__(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        chunk_indices = np.arange(start=0, stop=len(data), step=self.update_period)
        chunks = np.split(data, chunk_indices[1:])
        out = np.empty(len(data))
        frag_gain_interp = np.empty(len(data))

        for idx, chunk in zip(chunk_indices, chunks):
            power = np.mean(np.abs(chunk) ** 2)
            if np.isnan(power):
                out[idx:idx + self.update_period] = np.nan
                gain_new = self.gain * self.decay_step_size
                frag_gain_interp[idx:idx + self.update_period] = np.linspace(self.gain, gain_new, len(chunk))
                self.gain = gain_new
            else:
                error = self.ref - (np.log10(power) * 10.0 + 2.0 * self.gain)
                gain_new = self.gain + self.step_size * error
                gain_new = min(gain_new, self.gain_max)
                gain_interp = np.linspace(self.gain, gain_new, len(chunk))
                out[idx:idx + self.update_period] = chunk * (10.0 ** (gain_interp / 10.0))
                frag_gain_interp[idx:idx + self.update_period] = gain_interp
                self.gain = gain_new

        self.frag_gain_interp = frag_gain_interp

        if target is not None:
            return target * (10 ** (frag_gain_interp / 10.0))
        else:
            return out
