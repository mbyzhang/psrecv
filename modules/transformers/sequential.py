import functools
from typing import List
from modules.transformers import Transformer

class Sequential(Transformer):
    def __init__(self, *args: Transformer) -> None:
        super().__init__()
        self.children = args

    def __call__(self, data = None):
        return functools.reduce(lambda v, e: e(v), self.children, data)
