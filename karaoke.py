from lyrics_aligner import LyricsAligner
from lyrics_extractor import LyricsExtractor
from music_player import MusicPlayer, PyGameMusicPlayer
from un_mix_model import UnMixModel, Demucs3
from speech_recognition_model import GoogleRecognizer, SpeechRecognitionModel
import os
import threading


class Karaoke:
    def __init__(self, files_path: str, music_player: MusicPlayer = PyGameMusicPlayer,
                 un_mix_model: UnMixModel = Demucs3,
                 speech_recognition_model: SpeechRecognitionModel = GoogleRecognizer):
        self.files_path = files_path
        self.music_list = self.__get_music_list()
        self.un_mix_model = un_mix_model()
        self.recognition_model = speech_recognition_model()
        self.music_player = music_player()
        self.vocal_audio_path = None
        self.no_vocal_audio_path = None

    def start_karaoke(self, selected_music_name: str):
        music_path = os.path.join(self.files_path, selected_music_name)
        self.vocal_audio_path, self.no_vocal_audio_path = self.__separate_tracks(music_path)
        self.__play_music(self.no_vocal_audio_path)
        self.__show_lyrics()

    def __show_lyrics(self):
        le = LyricsExtractor()
        lyric = le.extract()
        print(lyric)
        aligned_words = LyricsAligner().align(lyric, self.vocal_audio_path)
        print(aligned_words)

    def __get_music_list(self):
        return os.listdir(self.files_path)

    def __play_music(self, music_path: str):
        x = threading.Thread(target=self.music_player.play, args=(music_path,))
        x.start()

    def __separate_tracks(self, audio_path: str):
        return self.un_mix_model.predict(audio_path)
