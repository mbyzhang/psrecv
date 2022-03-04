import functools
from typing import List
from modules.transformers import Transformer

class Sequential(Transformer):
    def __init__(self, *args: List[Transformer]) -> None:
        super().__init__()
        self.children = args

    def accept(self, data = None):
        return functools.reduce(lambda v, e: e.accept(v), self.children, data)
