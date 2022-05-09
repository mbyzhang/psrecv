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

    num_samples = sig.shape[-1]
    symbols = np.array([], dtype=float).reshape(sig.shape[:-1] + (0,))
    r = sps / (np.pi * 2.0)

    def get_symbols(phase, sps, lower, upper):
        start = ((phase - phase_initial) * r + sps // 2 - lower) % sps + lower
        symbol_idxs = np.arange(start=start, stop=upper, step=sps, dtype=int)
        symbols = np.zeros(sig.shape[:-1] + (len(symbol_idxs),), dtype=float)
        for i, symbol_idx in enumerate(symbol_idxs):
            start = max(0, symbol_idx - median_window_size // 2)
            end = symbol_idx + median_window_size // 2 + 1
            symbols[..., i] = np.median(sig[..., start:end], axis=-1)
        return symbols
    
    next_idx = num_samples if len(symbol_rising_edges_idxs) == 0 else symbol_rising_edges_idxs[0]
    symbols = np.concatenate((symbols, get_symbols(phase, sps, 0, next_idx)), axis=-1)

    for i, idx in enumerate(symbol_rising_edges_idxs):
        if i == len(symbol_rising_edges_idxs) - 1:
            next_idx = num_samples
        else:
            next_idx = symbol_rising_edges_idxs[i + 1]
        
        symbol_phase = (idx + phase_initial * r) % sps / r
        phase += phase_diff(symbol_phase, phase) * alpha
        symbols = np.concatenate((symbols, get_symbols(phase, sps, idx, next_idx)), axis=-1)
        
    return phase, symbols
