from numpy import block
from . import SoundSource
import soundfile

class SoundFileSource(SoundSource):
    def __init__(self, filename, block_size=4096):
        self.__file = soundfile.SoundFile(filename)
        self.__blocks_size = block_size
    
    @property
    def fs(self):
        return self.__file.samplerate
    
    @property
    def stream(self):
        return self.__blocks
    
    def __enter__(self):
        self.__file.seek(0)
        self.__blocks = self.__file.blocks(blocksize=self.__blocks_size)
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        if hasattr(self, "__file"):
            self.__file.close()
