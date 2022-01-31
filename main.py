#!/usr/bin/env python3

import numpy as np
import logging

from cdr import ClockDataRecoverer
from deframer import Deframer
from demodulator import Demodulator
from audioio import AudioRecordingDeviceSource, SoundFileSource

logging.basicConfig(level=logging.INFO)

def process_block(data: np.ndarray, demodulator: Demodulator, cdr: ClockDataRecoverer, deframer: Deframer):
    f1_f2_ratios = demodulator.accept(data)
    symbols = cdr.accept(f1_f2_ratios)
    frames = deframer.accept(symbols)
    for frame in frames:
        print(frame)

if __name__ == "__main__":
    block_size = 4096
    filename = "../Samples/100bps_frame.wav"
    baudrate = 100

    # source = SoundFileSource(filename, block_size)
    source = AudioRecordingDeviceSource(block_size=block_size)

    with source:
        fs = source.fs
        sps = fs // baudrate

        demodulator = Demodulator(
            fs=fs,
            f1=3000,
            f2=3200,
            f_delta=100,
            carrier_bandpass_ntaps=1229,
            symbol_lpf_cutoff_freq=1100,
            symbol_lpf_ntaps=405,
            eps=1e-6,
        )

        cdr = ClockDataRecoverer(
            sps=sps,
            clk_recovery_window=sps // 4,
            clk_recovery_grad_threshold=0.03,
            median_window_size=int(sps * 0.8)
        )

        deframer = Deframer()

        for block in source.stream:
            # print(block)
            process_block(block, demodulator, cdr, deframer)

