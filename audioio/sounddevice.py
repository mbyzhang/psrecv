import sounddevice
from . import SoundSource

class SoundDeviceSource(SoundSource):
    def __init__(self, device=None, fs=48000, block_size=4096):
        self.__fs = fs
        self.__block_size = block_size
        self.__stream = sounddevice.InputStream(fs, block_size, device, channels=1, dtype='float32')
    
    @property
    def fs(self):
        return self.__fs
    
    @property
    def stream(self):
        while True:
            yield self.__stream.read(self.__block_size)[0][:,0]
    
    def __enter__(self):
        self.__stream.__enter__()

    def __exit__(self, type, value, traceback):
        self.__stream.__exit__()
