from dsp import DSPBlock
from dsp.filters import FIRFilter, EMAFilter

import numpy as np

class CarrierDetectorExtractor(DSPBlock):
    def __init__(
        self,
        fs,
        carrier_cutoff,
        noise_delta, # noise bandwidth
        ntaps,
        noise_guard_bw=100.0, # guard bandwidth
        update_period=512,
        snr_threshold_db=6,
        ema_alpha=0.7,
    ) -> None:
        super().__init__()
        carrier_cutoff = np.array(carrier_cutoff, dtype=float)
        self.carrier_span = np.sum(carrier_cutoff[1::2] - carrier_cutoff[0::2])
        carrier_min_f = min(carrier_cutoff)
        carrier_max_f = max(carrier_cutoff)
        self.noise_span = noise_delta * 2

        self.carrier_filter = FIRFilter(fs=fs, cutoff=carrier_cutoff, ntaps=ntaps, pass_zero=False)

        self.noise_filter = FIRFilter(
            fs=fs, 
            cutoff=[
                carrier_min_f - noise_guard_bw - noise_delta,
                carrier_min_f - noise_guard_bw,
                carrier_max_f + noise_guard_bw,
                carrier_max_f + noise_guard_bw + noise_delta
            ],
            ntaps=ntaps,
            pass_zero=False
        )

        self.snr_lpf = EMAFilter(alpha=ema_alpha)

        self.update_period = update_period
        self.snr_threshold_db = snr_threshold_db

        self.frag_snr_db = None

    def __call__(self, data: np.ndarray) -> np.ndarray:
        chunk_indices = np.arange(start=0, stop=len(data), step=self.update_period)
        chunks = np.split(data, chunk_indices[1:])
        out = np.empty(len(data))
        frag_snr_db = np.empty(len(data))

        for idx, chunk in zip(chunk_indices, chunks):
            carrier = self.carrier_filter(chunk)
            noise = self.noise_filter(chunk)
            carrier_energy = np.mean(np.abs(carrier) ** 2)
            noise_energy = np.mean(np.abs(noise) ** 2)
            snr_db = 10 * (np.log10(carrier_energy) - np.log10(noise_energy / self.noise_span * self.carrier_span))
            snr_db = self.snr_lpf(snr_db)

            frag_snr_db[idx:idx + self.update_period] = snr_db

            if snr_db >= self.snr_threshold_db:
                out[idx:idx + self.update_period] = carrier
            else:
                out[idx:idx + self.update_period] = np.nan

        self.frag_snr_db = frag_snr_db

        return out
