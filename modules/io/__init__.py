class SoundSource:
    @property
    def stream(self):
        raise NotImplementedError()

from .sounddevice import SoundDeviceSource
from .soundfile import SoundFileSource
