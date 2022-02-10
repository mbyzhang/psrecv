import numpy as np

def symbol_rising_edge_detect(ratios, window_size, grad_threshold):
    zero_crossings = np.where(np.diff(np.sign(ratios)) > 0)[0]

    ratios_padded = np.pad(ratios, window_size // 2)
    grads = (ratios_padded[zero_crossings + (window_size // 2 * 2)] - \
                              ratios_padded[zero_crossings]) / window_size
    
    symbol_rising_edges_idxs = zero_crossings[grads > grad_threshold]

    return symbol_rising_edges_idxs

def clock_recovery(sig, symbol_rising_edges_idxs, sps, phase=0.0, phase_initial=0.0, alpha=0.002, median_window_size=1):
    def phase_diff(a, b):
        return (a - b + np.pi) % (np.pi * 2.0) - np.pi

    symbols = np.array([], dtype=float)
    r = sps / (np.pi * 2.0)

    def get_symbols(phase, sps, lower, upper):
        start = ((phase - phase_initial) * r + sps // 2 - lower) % sps + lower
        symbol_idxs = np.arange(start=start, stop=upper, step=sps, dtype=int)
        symbols = np.zeros(len(symbol_idxs), dtype=float)
        for i, symbol_idx in enumerate(symbol_idxs):
            start = max(0, symbol_idx - median_window_size // 2)
            end = symbol_idx + median_window_size // 2 + 1
            symbols[i] = np.median(sig[start:end])
        return symbols
    
    next_idx = len(sig) if len(symbol_rising_edges_idxs) == 0 else symbol_rising_edges_idxs[0]
    symbols = np.concatenate((symbols, get_symbols(phase, sps, 0, next_idx)))

    for i, idx in enumerate(symbol_rising_edges_idxs):
        if i == len(symbol_rising_edges_idxs) - 1:
            next_idx = len(sig)
        else:
            next_idx = symbol_rising_edges_idxs[i + 1]
        
        symbol_phase = (idx + phase_initial * r) % sps / r
        phase += phase_diff(symbol_phase, phase) * alpha
        symbols = np.concatenate((symbols, get_symbols(phase, sps, idx, next_idx)))
        
    return phase, symbols

class ClockDataRecoverer():
    def __init__(self, sps, clk_recovery_window, clk_recovery_grad_threshold=0.03, median_window_size=1):
        self.sps = sps
        self.clk_recovery_window = clk_recovery_window
        self.clk_recovery_grad_threshold = clk_recovery_grad_threshold
        self.symbol_phase = 0.0
        self.first_symbol_offset = 0
        self.median_window_size = median_window_size

    def accept(self, f1_f2_ratios):
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

        symbols_hard = np.where(symbols_soft >= 0, False, True)

        return symbols_hard
