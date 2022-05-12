from dsp import DSPBlock, Delay

import numpy as np

class DifferentialDecoder(DSPBlock):
    def __init__(self) -> None:
        super().__init__()
        self.delay = Delay(1, dtype=bool)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        self.stat_out = self.delay(data) != data
        return self.stat_out
