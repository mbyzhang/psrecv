from dataclasses import dataclass, field
from scipy import signal
from typing import List
import numpy as np
from enum import Enum
from encdec8b10b import EncDec8B10B
from reedsolo import RSCodec
from modules.transformers import Transformer

import logging
import math

logger = logging.getLogger(__name__)

class Deframer(Transformer):
    SFD_SYMBOL = np.unpackbits(
        np.array([EncDec8B10B.enc_8b10b(0x3c, 0, 1)[1]], dtype='uint16').view('uint8'),
        bitorder = 'little'
    )[:10]
    SYMBOLS_KEEP_LENGTH = 10

    class StateType(Enum):
        SEARCHING = 0
        IN_HEADER = 1
        IN_PAYLOAD = 2

    class FormatType(Enum):
        STANDARD = 0
        RAW_PAYLOAD = 1

    @dataclass
    class RSBuffer:
        len_target: int = 0
        len_parity: int = 0
        buf: bytearray = field(default_factory=bytearray)
        erase_pos: List[int] = field(default_factory=list)

    def __init__(self, payload_parity_len_ratio=0.2, format: FormatType = FormatType.STANDARD):
        self.payload_parity_len_ratio = payload_parity_len_ratio
        self.symbols = np.array([], dtype=bool)
        self.state = Deframer.StateType.SEARCHING
        self.format = format
        self.rs_buffer = self.RSBuffer()

    def __call__(self, symbols_in: np.ndarray) -> List[bytes]:
        symbols_in = np.array(symbols_in, dtype=bool)
        self.symbols = np.concatenate((self.symbols, symbols_in))
        out = []

        while True:
            if self.state == Deframer.StateType.SEARCHING:
                if len(self.symbols) == 0:
                    break
                corr = signal.correlate(np.where(self.symbols, 1, -1), np.where(Deframer.SFD_SYMBOL, 1, -1), mode="valid")
                sfd_ends_idx = np.where(corr >= 8)[0]
                
                if len(sfd_ends_idx) > 0:
                    logger.debug(f"Found SFD starting at {sfd_ends_idx}")

                    self.symbols = self.symbols[sfd_ends_idx[0] + 10:]
                    self.state = Deframer.StateType.IN_HEADER
                    self.rs_buffer = self.RSBuffer(len_target=3, len_parity=2)
                else:
                    break
            elif self.state == Deframer.StateType.IN_HEADER or (self.state == self.StateType.IN_PAYLOAD and self.format == self.FormatType.STANDARD):
                if len(self.symbols) < 10:
                    break
                symbol, self.symbols = np.packbits(self.symbols[:10], bitorder='little').view('uint16').item(), self.symbols[10:]
                logger.debug(f"Consumed word {symbol:03x}")

                byte = 0xff
                try:
                    ctrl, byte = EncDec8B10B.dec_8b10b(symbol)
                    if ctrl:
                        byte = 0xff
                        raise Exception(f"Unexpected control word: {byte:02x}")
                except Exception as e:
                    self.rs_buffer.erase_pos.append(len(self.rs_buffer.buf))
                    logger.warning(e)
                    if len(self.rs_buffer.erase_pos) > self.rs_buffer.len_parity:
                        logger.warning("Stop receiving early due to too many errors")
                        self.state = self.StateType.SEARCHING
                        self.rs_buffer = None
                        break
                
                self.rs_buffer.buf.append(byte)
                
                if len(self.rs_buffer.buf) == self.rs_buffer.len_target:
                    rsc = RSCodec(self.rs_buffer.len_parity, fcr=1)
                    
                    try:
                        logger.debug(f"Header: {self.rs_buffer}")
                        buf_decoded = rsc.decode(self.rs_buffer.buf, erase_pos=self.rs_buffer.erase_pos)[0]
                    except:
                        logger.warning("Failed to decode frame due to too many errors")
                        self.state = self.StateType.SEARCHING
                        self.rs_buffer = None
                    
                    if self.state == self.StateType.IN_HEADER:
                        assert len(buf_decoded) == 1
                        payload_decoded_len = buf_decoded[0]
                        len_parity = math.ceil(payload_decoded_len * self.payload_parity_len_ratio)

                        self.state = self.StateType.IN_PAYLOAD

                        if self.format == self.FormatType.STANDARD:
                            self.rs_buffer = self.RSBuffer(
                                len_parity=len_parity,
                                len_target=payload_decoded_len + len_parity
                            )
                        else:
                            self.rs_buffer = self.RSBuffer(len_target=payload_decoded_len)

                        logger.info(f"Receiving message with length {payload_decoded_len}")

                    elif self.state == self.StateType.IN_PAYLOAD:
                        out.append(buf_decoded)
                        self.state = self.StateType.SEARCHING
                        self.rs_buffer = None

                        logger.info(f"Received message {buf_decoded}")
            elif self.state == self.StateType.IN_PAYLOAD and self.format == self.FormatType.RAW_PAYLOAD:
                if len(self.symbols) < 8:
                    break
                symbol, self.symbols = np.packbits(self.symbols[:8], bitorder='little')[0], self.symbols[8:]
                buf = self.rs_buffer.buf
                buf.append(symbol)
                logger.debug(f"Got byte 0x{symbol:02x}")

                if len(buf) == self.rs_buffer.len_target:
                    out.append(bytes(buf))

                    self.state = self.StateType.SEARCHING
                    self.rs_buffer = None
                    logger.info(f"Received raw message {buf}")
            else:
                assert False
        
        self.symbols = self.symbols[-Deframer.SYMBOLS_KEEP_LENGTH:]
        return out
