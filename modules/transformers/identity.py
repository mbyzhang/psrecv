from modules.transformers import Transformer

class Identity(Transformer):
    def __call__(self, data=None):
        return data
