from modules.transformers import Transformer
from modules.transformers.filters import FIRFilter, EMAFilter

import numpy as np

class CarrierDetectorExtractor(Transformer):
    def __init__(
        self,
        fs,
        carrier_cutoff,
        noise_delta,
        ntaps,
        update_period=512,
        snr_threshold_db=6,
        ema_alpha=0.7,
    ) -> None:
        super().__init__()
        carrier_min_f, carrier_max_f = min(carrier_cutoff), max(carrier_cutoff)

        self.carrier_filter = FIRFilter(fs=fs, cutoff=carrier_cutoff, ntaps=ntaps, pass_zero=False)

        self.noise_filter = FIRFilter(
            fs=fs, 
            cutoff=[
                carrier_min_f - noise_delta, 
                carrier_min_f,
                carrier_max_f,
                carrier_max_f + noise_delta
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

            snr_db = 10 * (np.log10(np.mean(np.abs(carrier) ** 2)) - np.log10(np.mean(np.abs(noise) ** 2)))
            snr_db = self.snr_lpf(snr_db)

            frag_snr_db[idx:idx + self.update_period] = snr_db

            if snr_db >= self.snr_threshold_db:
                out[idx:idx + self.update_period] = carrier
            else:
                out[idx:idx + self.update_period] = np.nan

        self.frag_snr_db = frag_snr_db

        return out
