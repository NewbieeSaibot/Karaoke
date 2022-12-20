from abc import ABC
from typing import Tuple
import numpy as np
import os


class UnMixModel(ABC):
    def __init__(self):
        pass

    def predict(self, audio: np.ndarray):
        pass


class Demucs3(UnMixModel):
    def __init__(self):
        super().__init__()

    def predict(self, audio_path: str) -> Tuple[str, str]:

        audio_name = audio_path.split("\\")[-1].split(".")[0]
        base_path = "C:\\Users\\tobia\\Desktop\\karaoke\\separated\\mdx_extra_q"
        if os.path.exists(os.path.join(base_path, audio_name)):
            return (os.path.join(base_path, audio_name, "vocals.wav"),
                    os.path.join(base_path, audio_name, "other.wav"))

        self.run_model(audio_path)
        return (os.path.join(base_path, audio_name, "vocals.wav"),
                os.path.join(base_path, audio_name, "other.wav"))

    def run_model(self, audio_path: str):
        print(f"python -m demucs3 \"{audio_path}\"")
        os.system(f"python -m demucs3 \"{audio_path}\"")
