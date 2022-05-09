from dsp import DSPBlock

# Expoential moving average filter
class EMAFilter(DSPBlock):
    def __init__(
        self,
        alpha=0.99,
        s_initial=0.0,
    ) -> None:
        super().__init__()
        self.s = s_initial
        self.alpha = alpha

    def __call__(self, data: float) -> float:
        self.s = self.alpha * self.s + (1.0 - self.alpha) * data
        return self.s
