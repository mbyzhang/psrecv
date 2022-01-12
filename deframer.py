

from utils import int_to_bin, np_bin_array_to_int
from scipy import signal
from typing import List
import numpy as np
from enum import Enum
from encdec8b10b import EncDec8B10B

import logging

logger = logging.getLogger(__name__)

class Deframer():
    SFD_SYMBOL = int_to_bin(EncDec8B10B.enc_8b10b(0x3c, 0, 1)[1])
    EFD = 0x1c
    MFS = 64 # Maximum frame size
    SYMBOLS_KEEP_LENGTH = 10

    class StateType(Enum):
        SEARCHING = 0
        IN_FRAME = 1

    def __init__(self):
        self.symbols = np.array([], dtype=bool)
        self.state = Deframer.StateType.SEARCHING
        self.frame_rx_bytes = 0
        self.frame = bytearray()

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
                    self.state = Deframer.StateType.IN_FRAME
                    self.frame_rx_bytes = 0
                    self.frame = bytearray()
                else:
                    break
            elif self.state == Deframer.StateType.IN_FRAME:
                if len(self.symbols) < 10:
                    break
                symbol, self.symbols = np_bin_array_to_int(self.symbols[:10]), self.symbols[10:]
                logger.debug(f"Consumed word {symbol:03x}")

                try:
                    ctrl, byte = EncDec8B10B.dec_8b10b(symbol)
                    if ctrl:
                        if byte == Deframer.EFD:
                            logger.debug("End of frame detected")
                            out.append(self.frame)
                            self.state = Deframer.StateType.SEARCHING
                            self.frame_rx_bytes = 0
                            self.frame = bytearray()
                            continue
                        else:
                            raise Exception(f"Unexpected control word: {byte:02x}")
                    else:
                        self.frame.append(byte)
                        logger.debug(f"Output byte {byte:02x}")

                except Exception as e:
                    self.frame.append(0xff)
                    logger.warning(e)

                self.frame_rx_bytes += 1
                if self.frame_rx_bytes > Deframer.MFS:
                    logger.debug(f"Frame too long, ignoring")
                    self.state = Deframer.StateType.SEARCHING
                    self.frame_rx_bytes = 0
                    self.frame = bytearray()
        
        self.symbols = self.symbols[-Deframer.SYMBOLS_KEEP_LENGTH:]
        return out
