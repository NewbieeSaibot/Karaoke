from music_player import MusicPlayer, PyGameMusicPlayer
from un_mix_model import UnMixModel, Demucs3
import os
import threading


class Karaoke:
    def __init__(self, files_path: str, music_player: MusicPlayer = PyGameMusicPlayer,
                 un_mix_model: UnMixModel = Demucs3):
        self.files_path = files_path
        self.music_list = self.__get_music_list()
        self.un_mix_model = un_mix_model()
        self.recognition_model = None  # SpeechRecognition
        self.music_player = music_player()

    def start_karaoke(self, selected_music_name: str):
        self.__play_others(selected_music_name)
        self.__show_lyrics()

    def __show_lyrics(self):
        pass

    def __play_others(self, music_name: str):
        music_path = os.path.join(self.files_path, music_name)
        vocal_audio_path, other_audio_path = self.__separate_tracks(music_path)
        self.__play_music(other_audio_path)

    def __get_music_list(self):
        return os.listdir(self.files_path)

    def __play_music(self, music_path: str):
        x = threading.Thread(target=self.music_player.play, args=(music_path,))
        x.start()

    def __separate_tracks(self, audio_path: str):
        return self.un_mix_model.predict(audio_path)
