from dsp import DSPBlock

class Sequential(DSPBlock):
    def __init__(self, *args: DSPBlock) -> None:
        super().__init__()
        self.children = args
        self.outputs = [None] * len(self.children)

    def __call__(self, data = None):
        v = data

        for i, child in enumerate(self.children):
            v = child(v)
            self.outputs[i] = v

        return v
