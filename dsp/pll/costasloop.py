from dsp import DSPBlock
from dsp.filters import FIRFilter

import numpy as np

# Order-2 Costas loop
class CostasLoop(DSPBlock):
    def __init__(
        self,
        f,
        fs,
        lpf_cutoff=1100,
        lpf_ntaps=405,
        update_period=512,
        alpha=0.1,
        beta=0.01,
    ) -> None:
        super().__init__()

        self.update_period = update_period
        self.freq = f / fs * np.pi * 2.0
        self.phase = 0.0
        self.alpha = alpha
        self.beta = beta
        self.fs = fs

        self.lpf_i = FIRFilter(
            fs=fs,
            cutoff=lpf_cutoff,
            ntaps=lpf_ntaps,
            pass_zero=True
        )

        self.lpf_q = FIRFilter(
            fs=fs,
            cutoff=lpf_cutoff,
            ntaps=lpf_ntaps,
            pass_zero=True
        )

    def __call__(self, data):
        chunk_indices = np.arange(start=0, stop=len(data), step=self.update_period)
        chunks = np.split(data, chunk_indices[1:])
        stat_phase = np.full(len(chunks), np.nan)
        stat_freq = np.full(len(chunks), np.nan)
        stat_e_raw = np.full(len(chunks), np.nan)
        out = np.full(len(data), np.nan)

        for i, (idx, chunk) in enumerate(zip(chunk_indices, chunks)):
            p = chunk * np.exp(-1j * (self.freq * np.arange(len(chunk)) + self.phase))
            i_amp = self.lpf_i(np.real(p))
            q_amp = self.lpf_q(np.imag(p))
            iq_amp = np.sqrt(i_amp ** 2 + q_amp ** 2)
            e = np.mean(i_amp * q_amp)
            stat_e_raw[i] = e
            e = np.clip(e, -0.5, 0.5)
            if np.isnan(e):
                self.phase = 0.0
                continue

            self.freq += self.beta * e
            self.phase += self.freq * len(chunk) + self.alpha * e
            self.phase %= np.pi * 2.0

            stat_phase[i] = self.phase
            stat_freq[i] = self.freq
            out[idx:idx + self.update_period] = i_amp / iq_amp

        self.stat_phase = stat_phase
        self.stat_freq = stat_freq
        self.stat_e_raw = stat_e_raw
        return out
