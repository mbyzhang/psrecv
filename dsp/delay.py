from dsp import DSPBlock
import numpy as np

class Delay(DSPBlock):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.padding = np.zeros(self.n)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return data

        x = np.concatenate((self.padding, data))
        self.padding = x[-self.n:]
        return x[:len(data)]
