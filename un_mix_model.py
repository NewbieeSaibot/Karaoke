from abc import ABC
from typing import Tuple
import numpy as np
import os
import librosa
import soundfile


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
                    os.path.join(base_path, audio_name, "no_vocal.wav"))

        self.run_model(audio_path)
        self.build_no_vocals(os.path.join(base_path, audio_name))
        return (os.path.join(base_path, audio_name, "vocals.wav"),
                os.path.join(base_path, audio_name, "no_vocal.wav"))

    @staticmethod
    def run_model(audio_path: str):
        print(f"python -m demucs3 \"{audio_path}\"")
        os.system(f"python -m demucs3 \"{audio_path}\"")

    @staticmethod
    def build_no_vocals(audios_path: str):
        other, _ = librosa.load(os.path.join(audios_path, "other.wav"))
        drums, _ = librosa.load(os.path.join(audios_path, "drums.wav"))
        bass, sr = librosa.load(os.path.join(audios_path, "bass.wav"))
        no_vocal = other + drums + bass
        soundfile.write(os.path.join(audios_path, "no_vocal.wav"), no_vocal, sr)
