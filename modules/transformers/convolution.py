from scipy import signal
from modules.transformers import Transformer
import numpy as np

class StatefulConvolution(Transformer): # for processing continuous signal in chunks. Any better ways?
    def __init__(self, fir_filter):
        super().__init__()
        self.last_sample = None
        self.fir_filter = fir_filter
        self.ntaps = len(fir_filter)
    
    def accept(self, sample):
        return self.convolve(sample)
    # PRECONDITION: len(sample) >= self.ntaps - 1
    def convolve(self, sample):
        if self.last_sample is None:
            self.last_sample = np.zeros(len(sample))
        
        # assert len(self.last_sample) >= self.ntaps - 1
        
        padding = self.last_sample[-self.ntaps + 1:]
        x = np.concatenate((padding, sample))
        self.last_sample = sample
        return signal.convolve(x, self.fir_filter, mode='valid')
