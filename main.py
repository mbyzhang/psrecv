#!/usr/bin/env python3

import numpy as np
import logging
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description='PSRecv')
    parser.add_argument("-d", "--device", help="Audio input device name")
    parser.add_argument("-i", "--input", help="Input audio file name")
    parser.add_argument("-b", "--baudrate", help="Baudrate", default=100, type=int)
    parser.add_argument("-r", "--sample-rate", help="Sample rate", default=48000, type=int)
    parser.add_argument("--block-size", help="Block size", default=4096, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    block_size = args.block_size
    baudrate = args.baudrate

    if args.input:
        filename = args.input
        source = SoundFileSource(filename, block_size=block_size)
    else:
        device = args.device
        sample_rate = args.sample_rate
        source = AudioRecordingDeviceSource(device, fs=sample_rate, block_size=block_size)

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

