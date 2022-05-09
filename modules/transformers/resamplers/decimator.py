from modules.transformers import Transformer
from modules.transformers.filters import FIRFilter

import numpy as np

class Decimator(Transformer):
    def __init__(self, down=8) -> None:
        super().__init__()

        # anti-aliasing filter
        self.filter = FIRFilter(
            fs=2.0,
            cutoff=1.0 / down,
            ntaps=20 * down + 1,
            pass_zero=True
        )

        self.q = down

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.filter(data)[::self.q]
