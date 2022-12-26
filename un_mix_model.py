from abc import ABC
from typing import Tuple
import numpy as np
import os
import librosa
import soundfile


class UnMixModel(ABC):
    def __init__(self, base_path: str):
        self.base_path = base_path

    def predict(self, audio: np.ndarray):
        pass


class Demucs3(UnMixModel):
    def __init__(self, base_path: str):
        super().__init__(base_path)

    def predict(self, music_name: str) -> Tuple[str, str]:
        separated_files_path = os.path.join(os.getcwd(), "separated", "mdx_extra_q")
        if os.path.exists(os.path.join(separated_files_path, music_name)):
            return (os.path.join(separated_files_path, music_name, "vocals.wav"),
                    os.path.join(separated_files_path, music_name, "no_vocal.wav"))

        self.run_model(music_name)
        self.build_no_vocals(os.path.join(separated_files_path, music_name))
        return (os.path.join(separated_files_path, music_name, "vocals.wav"),
                os.path.join(separated_files_path, music_name, "no_vocal.wav"))

    def run_model(self, music_name: str):
        music_path = os.path.join(self.base_path, music_name, music_name + ".mp3")
        print(f"python -m demucs3 \"{music_path}\"")
        os.system(f"python -m demucs3 \"{music_path}\"")

    @staticmethod
    def build_no_vocals(audios_path: str):
        other, _ = librosa.load(os.path.join(audios_path, "other.wav"))
        drums, _ = librosa.load(os.path.join(audios_path, "drums.wav"))
        bass, sr = librosa.load(os.path.join(audios_path, "bass.wav"))
        no_vocal = other + drums + bass
        soundfile.write(os.path.join(audios_path, "no_vocal.wav"), no_vocal, sr)
