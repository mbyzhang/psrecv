from dsp import DSPBlock
from .carrierdetext import CarrierDetectorExtractor
from .logloopagc import LogLoopAGC

import numpy as np

class GeneralPreprocessor(DSPBlock):
    def __init__(self,
        fs,
        carrier_freqs,
        carrier_f_delta,
        agc_ref = -3.0,
        agc_gain_max = 50.0,
        det_snr_threshold_db = 10.0,
        no_agc: bool = False
    ):
        carrier_freqs = np.array(carrier_freqs)
        carrier_cutoff = list(zip(carrier_freqs - carrier_f_delta, carrier_freqs + carrier_f_delta))
        carrier_cutoff = np.array(carrier_cutoff).flatten()

        # remove non-increasing parts in carrier_cutoff
        indices = np.where(np.diff(carrier_cutoff) <= 0)[0]
        indices = np.concatenate((indices, indices + 1))
        carrier_cutoff = np.delete(carrier_cutoff, indices, None)
        print(f"Carrier cutoff = {carrier_cutoff}")

        if not no_agc:
            self.agc = LogLoopAGC(
                step_size=0.2,
                ref=-agc_ref,
                gain_max=agc_gain_max,
                update_period=512,
            )

        self.sigdet = CarrierDetectorExtractor(
            fs=fs,
            carrier_cutoff=carrier_cutoff,
            noise_delta=carrier_f_delta,
            ntaps=1229,
            ema_alpha=0.6,
            snr_threshold_db=det_snr_threshold_db,
            update_period=512,
        )

        self.no_agc = no_agc

    def __call__(self, data: np.ndarray) -> np.ndarray:
        x_carrier = self.sigdet(data)
        if self.no_agc:
            x_out = data
        else:
            x_out = self.agc(x_carrier, data)
        x_out *= np.where(np.isnan(x_carrier), np.nan, 1.0)
        return x_out
