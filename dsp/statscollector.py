from typing import Callable, Dict, List
from dsp import DSPBlock

import numpy as np

class StatisticsCollector(DSPBlock):
    def __init__(self, block: DSPBlock, fields: Dict[str, Callable[[], np.ndarray]]) -> None:
        super().__init__()
        self.fields = fields
        self.block = block
        self.cum_data = dict()

    def __call__(self, data=None):
        out = self.block(data)
        for field_name, value_getter in self.fields.items():
            if field_name not in self.cum_data:
                self.cum_data[field_name] = value_getter()
            else:
                self.cum_data[field_name] = np.concatenate((self.cum_data[field_name], value_getter()), axis=-1)
        return out
