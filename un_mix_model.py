import numpy as np
from demucs3.hdemucs import HDemucs
from abc import ABC


class UnMixModel(ABC):
    def __init__(self):
        pass

    def predict(self, audio: np.ndarray):
        pass


class Demucs3(UnMixModel):
    def __init__(self):
        pass

    def predict(self, audio_window: np.ndarray):
        demucs3 = HDemucs()
        unmixed = demucs3.forward(audio_window)
        return unmixed
