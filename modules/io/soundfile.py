from . import SoundSource
import soundfile

class SoundFileSource(SoundSource):
    def __init__(self, filename, block_size=4096):
        self.__fs = soundfile.info(filename).samplerate
        self.__blocks = soundfile.blocks(filename, blocksize=block_size)
    
    @property
    def fs(self):
        return self.__fs
    
    @property
    def stream(self):
        return self.__blocks
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__blocks.close()
