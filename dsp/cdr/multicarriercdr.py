import numpy as np
from dsp import DSPBlock

from .common import symbol_rising_edge_detect, clock_recovery

class MulticarrierCDR(DSPBlock):
    def __init__(self, sps, clk_recovery_window, clk_recovery_grad_threshold=0.03, median_window_size=1):
        self.sps = sps
        self.clk_recovery_window = clk_recovery_window
        self.clk_recovery_grad_threshold = clk_recovery_grad_threshold
        self.symbol_phase = 0.0
        self.first_symbol_offset = 0
        self.median_window_size = median_window_size

    def __call__(self, sig_multicarrier: np.ndarray):
        assert sig_multicarrier.ndim == 2
        num_samples = sig_multicarrier.shape[1]

        symbol_rising_edges_idxs_multicarrier = [
            symbol_rising_edge_detect(
                sig, 
                self.clk_recovery_window, 
                self.clk_recovery_grad_threshold
            )
            for sig in sig_multicarrier
        ]

        symbol_rising_edges_idxs_combined = np.sort(np.concatenate(symbol_rising_edges_idxs_multicarrier))

        # for debugging
        symbol_rising_edges_multicarrier = np.zeros(sig_multicarrier.shape, dtype=np.bool8)
        for i, symbol_rising_edges_idxs in enumerate(symbol_rising_edges_idxs_multicarrier):
            symbol_rising_edges_multicarrier[i, symbol_rising_edges_idxs] = True
        self.symbol_rising_edges_multicarrier = symbol_rising_edges_multicarrier

        symbol_rising_edges_combined = np.zeros(num_samples, dtype=np.bool8)
        symbol_rising_edges_combined[symbol_rising_edges_idxs_combined] = True
        self.symbol_rising_edges_combined = symbol_rising_edges_combined
        
        self.symbol_phase, symbols_soft = clock_recovery(
            sig=sig_multicarrier, 
            symbol_rising_edges_idxs=symbol_rising_edges_idxs_combined, 
            sps=self.sps, 
            phase=self.symbol_phase, 
            phase_initial=self.first_symbol_offset / self.sps * np.pi * 2.0,
            alpha=0.1,
            median_window_size=self.median_window_size,
        )

        self.first_symbol_offset += num_samples
        self.first_symbol_offset %= self.sps

        symbols_hard = np.argmax(symbols_soft, axis=0)
        self.symbols_hard = symbols_hard
        return symbols_hard
