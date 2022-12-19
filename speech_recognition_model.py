import numpy as np
from abc import ABC


class SpeechRecognitionModel(ABC):
    def __init__(self):
        pass

    def predict(self, audio_window: np.ndarray):
        pass

