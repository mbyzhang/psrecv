

from dataclasses import dataclass
from utils import int_to_bin, np_bin_array_to_int
from scipy import signal
from typing import List
import numpy as np
from enum import Enum
from encdec8b10b import EncDec8B10B
from reedsolo import RSCodec

import logging
import math

logger = logging.getLogger(__name__)

class Deframer():
    SFD_SYMBOL = int_to_bin(EncDec8B10B.enc_8b10b(0x3c, 0, 1)[1])
    SYMBOLS_KEEP_LENGTH = 10

    class StateType(Enum):
        SEARCHING = 0
        IN_HEADER = 1
        IN_PAYLOAD = 2

    @dataclass
    class RSBuffer:
        len_target: int
        buf: bytearray
        erase_pos: List[int]

    def __init__(self, payload_parity_len_ratio=0.2):
        self.payload_parity_len_ratio = payload_parity_len_ratio
        self.symbols = np.array([], dtype=bool)
        self.state = Deframer.StateType.SEARCHING
        self.rs_buffer = self.RSBuffer()

    def accept(self, symbols_in: np.ndarray) -> List[bytearray]:
        symbols_in = np.array(symbols_in, dtype=bool)
        self.symbols = np.concatenate((self.symbols, symbols_in))
        out = []

        while True:
            if self.state == Deframer.StateType.SEARCHING:
                if len(self.symbols) == 0:
                    break
                corr = signal.correlate(np.where(self.symbols, 1, -1), np.where(Deframer.SFD_SYMBOL, 1, -1))
                sfd_ends_idx = np.where(corr == 10)[0]
                
                if len(sfd_ends_idx) > 0:
                    logger.debug(f"Found SFD ending at {sfd_ends_idx}")

                    self.symbols = self.symbols[sfd_ends_idx[0] + 1:]
                    self.state = Deframer.StateType.IN_HEADER
                    self.rs_buffer = self.RSBuffer(len_target=3)
                else:
                    break
            elif self.state in (Deframer.StateType.IN_HEADER, Deframer.StateType.IN_PAYLOAD):
                if len(self.symbols) < 10:
                    break
                symbol, self.symbols = np_bin_array_to_int(self.symbols[:10]), self.symbols[10:]
                logger.debug(f"Consumed word {symbol:03x}")

                try:
                    ctrl, byte = EncDec8B10B.dec_8b10b(symbol)
                    if ctrl:
                        byte = 0xff
                        raise Exception(f"Unexpected control word: {byte:02x}")
                except Exception as e:
                    self.rs_buffer.erase_pos.append(len(self.rs_buffer.buf))
                    logger.warning(e)
                
                self.rs_buffer.buf.append(byte)
                
                if len(self.rs_buffer.buf) == self.rs_buffer.len_target:
                    rsc = RSCodec(self.rs_buffer.len_target)
                    
                    try:
                        buf_decoded = rsc.decode(self.rs_buffer.buf, erase_pos=self.rs_buffer.erase_pos)[0]
                    except:
                        logging.warning("Failed to decode frame due to too many errors")
                        self.state = self.StateType.SEARCHING
                        self.rs_buffer = None
                    
                    if self.state == self.StateType.IN_HEADER:
                        assert len(buf_decoded) == 1
                        payload_decoded_len = buf_decoded[0]
                        self.state = self.StateType.IN_PAYLOAD
                        self.rs_buffer = self.RSBuffer(
                            len_target=payload_decoded_len + math.ceil(payload_decoded_len * self.payload_parity_len_ratio)
                        )

                        logging.info(f"Receiving message with length {payload_decoded_len}")

                    elif self.state == self.StateType.IN_PAYLOAD:
                        out.append(buf_decoded)
                        self.state = self.StateType.SEARCHING
                        self.rs_buffer = None

                        logging.info(f"Received message {buf_decoded}")
        
        self.symbols = self.symbols[-Deframer.SYMBOLS_KEEP_LENGTH:]
        return out
