#!/usr/bin/env python3

import logging
import argparse
import importlib
import sys

from modules.io import SoundDeviceSource, SoundFileSource

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description='PSRecv')
    parser.add_argument("-p", "--profile", help="Profile", default="bfsk_3khz")
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

    fs = source.fs
    sps = fs // baudrate

    try:
        pipeline = importlib.import_module("." + args.profile, "profiles").get_pipeline(fs, sps)
    except Exception as e:
        logger.critical(f"Cannot load {args.profile} profile: {e}")
        sys.exit(1)

    with source:
        for block in source.stream:
            frames = pipeline(block)
            for frame in frames:
                print(frames)
