from typing import Callable, Dict, List
from modules.transformers import Transformer

import numpy as np

class AuxDataCollector(Transformer):
    def __init__(self, block: Transformer, fields: Dict[str, Callable[[], np.ndarray]]) -> None:
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