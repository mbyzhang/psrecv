#!/usr/bin/env python3

import logging
import argparse

from modules.transformers import Sequential
from modules.transformers.cdr import SimpleCDR
from modules.transformers.framing import Deframer
from modules.transformers.demodulators import BFSKDemodulator
from modules.io import SoundDeviceSource, SoundFileSource

logging.basicConfig(level=logging.INFO)

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
        source = SoundDeviceSource(device, fs=sample_rate, block_size=block_size)

    with source:
        fs = source.fs
        sps = fs // baudrate

        pipeline = Sequential(
            BFSKDemodulator(
                fs=fs,
                f0=3200,
                f1=3000,
                f_delta=100,
                carrier_bandpass_ntaps=1229,
                symbol_lpf_cutoff_freq=1100,
                symbol_lpf_ntaps=405,
                eps=1e-6,
            ),
            SimpleCDR(
                sps=sps,
                clk_recovery_window=sps // 4,
                clk_recovery_grad_threshold=0.03,
                median_window_size=int(sps * 0.8)
            ),
            Deframer(
                format=Deframer.FormatType.STANDARD
            )
        )

        for block in source.stream:
            frames = pipeline(block)
            for frame in frames:
                print(frames)
