import numpy as np
from modules.transformers import Transformer

from .common import symbol_rising_edge_detect, clock_recovery

class SimpleCDR(Transformer):
    def __init__(self, sps, clk_recovery_window, clk_recovery_grad_threshold=0.03, median_window_size=1):
        self.sps = sps
        self.clk_recovery_window = clk_recovery_window
        self.clk_recovery_grad_threshold = clk_recovery_grad_threshold
        self.symbol_phase = 0.0
        self.first_symbol_offset = 0
        self.median_window_size = median_window_size

    def __call__(self, f1_f2_ratios):
        symbol_rising_edges_idxs = symbol_rising_edge_detect(
            f1_f2_ratios, 
            self.clk_recovery_window, 
            self.clk_recovery_grad_threshold
        )

        # for debugging
        symbol_rising_edges = np.zeros(len(f1_f2_ratios), dtype=np.bool8)
        symbol_rising_edges[symbol_rising_edges_idxs] = True
        self.symbol_rising_edges = symbol_rising_edges
        
        self.symbol_phase, symbols_soft = clock_recovery(
            sig=f1_f2_ratios, 
            symbol_rising_edges_idxs=symbol_rising_edges_idxs, 
            sps=self.sps, 
            phase=self.symbol_phase, 
            phase_initial=self.first_symbol_offset / self.sps * np.pi * 2.0,
            alpha=0.1,
            median_window_size=self.median_window_size,
        )

        self.first_symbol_offset += len(f1_f2_ratios)
        self.first_symbol_offset %= self.sps

        symbols_hard = np.where(symbols_soft >= 0, True, False)
        self.symbols_hard = symbols_hard
        return symbols_hard
