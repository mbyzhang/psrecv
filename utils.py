import numpy as np

def int_to_bin(x, digits=10, dtype=bool):
    ret = np.zeros(digits, dtype=dtype)
    for i in range(digits):
        ret[i] = (x >> i) & 0x1
    return ret

def np_bin_array_to_int(arr):
    return np.sum((2 ** np.arange(10)) * arr)
