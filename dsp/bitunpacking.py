import numpy as np
from dsp import DSPBlock

class BitUnpacking(DSPBlock):
    def __init__(self, count=8) -> None:
        self.count = count
        super().__init__()
    
    def __call__(self, data: np.ndarray):
        x = np.array(data, dtype=np.uint8)
        x = np.expand_dims(x, axis=-1)
        bits = np.unpackbits(x, axis=-1, count=self.count, bitorder='little')
        return bits.flatten()
