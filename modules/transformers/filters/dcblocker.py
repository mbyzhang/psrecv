from typing import Literal
from modules import transformers
from scipy import signal

import numpy as np

class DCBlocker(transformers.Transformer):
    def __init__(
        self,
        p: float = 0.995,
        s: int = 9600,
        mode: Literal["ma", "iir"] = "ma"
    ) -> None:
        super().__init__()

        if mode == "ma":
            self.b = np.concatenate(([1.0], np.full(s, -1.0/s)))
            self.a = [1.0]
        elif mode == "iir":
            self.b = [1.0, -1.0]
            self.a = [1.0, -p]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.zi = signal.lfiltic(self.b, self.a, [])
        self.z = self.zi

    def __call__(self, data: np.ndarray) -> np.ndarray:
        out, self.z = signal.lfilter(self.b, self.a, np.nan_to_num(data), zi=self.z)
        out = np.where(np.isnan(data), np.nan, out)
        self.frag_out = out
        return out

# Reference: https://www.iro.umontreal.ca/~mignotte/IFT3205/Documents/TipsAndTricks/DCBlockerAlgorithms.pdf
