from dsp import DSPBlock

class Identity(DSPBlock):
    def __call__(self, data=None):
        return data
